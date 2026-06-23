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

//! Ministral 3 model implementation using mlxcel-core
//!
//! Key features:
//! - layer_types: Vec<String> with "sliding_attention" or "full_attention"
//! - Llama 4 attention scaling per position: scale = 1 + beta * ln(1 + floor(pos / max_pos))
//! - RotatingKVCache for sliding_attention layers

use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask};
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
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    #[serde(default)]
    pub sliding_window: Option<usize>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub llama_4_scaling_beta: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl ModelArgs {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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

    pub fn layer_types_vec(&self) -> Vec<String> {
        self.layer_types
            .clone()
            .unwrap_or_else(|| vec!["full_attention".to_string(); self.num_hidden_layers])
    }
}

// Llama 4 Attention Scaling.
/// Compute Llama 4 attention scale per position
/// scale = 1 + beta * ln(1 + floor(pos / max_pos))
fn get_llama4_attn_scale(
    size: i32,
    offset: i32,
    beta: f32,
    max_position_embeddings: usize,
) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let pos = (i + offset) as f32;
            let max_pos = max_position_embeddings as f32;
            1.0 + beta * (1.0 + (pos / max_pos).floor()).ln()
        })
        .collect()
}

// Attention.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub window_size: i32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        attn_scale: &[f32],
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();

        // Apply RoPE
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Apply Llama 4 attention scaling
        // attn_scale: [seq_len] -> reshape to [1, 1, seq_len, 1] for broadcasting
        let scale_shape = vec![1, 1, l, 1];
        let scale_arr = mlxcel_core::from_slice_f32(attn_scale, &[l]);
        let scale_arr = mlxcel_core::reshape(&scale_arr, &scale_shape);
        let q = mlxcel_core::multiply(&q, &scale_arr);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention (handles GQA expansion internally)
        let attn_out = if l > 1 {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, self.window_size)
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let rope_base = args
            .rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .unwrap_or(10000.0);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale,
            rope_base,
            window_size: args.sliding_window.map(|w| w as i32).unwrap_or(0),
        })
    }
}

// MLP.
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // SiLU(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SiLU activation
        let gate_silu = mlxcel_core::silu(&gate);

        // Element-wise product
        let activated = mlxcel_core::multiply(&gate_silu, &up);

        // Down projection
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub use_sliding: bool,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        attn_scale: &[f32],
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, attn_scale, cache, mask);
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
        use_sliding: bool,
    ) -> Result<Self, String> {
        Self::from_weights_with_prefix(weights, args, layer_idx, use_sliding, "")
    }

    pub fn from_weights_with_prefix(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        use_sliding: bool,
        model_prefix: &str,
    ) -> Result<Self, String> {
        let prefix = format!("{}model.layers.{}", model_prefix, layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            use_sliding,
        })
    }
}

// Cache Interface.
pub trait CacheInterface {
    fn offset(&self) -> i32;
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

pub enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(c) => c,
            Cache::Rotating(c) => c,
        }
    }
}

// Ministral3 Model.
pub struct Ministral3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub sliding_window: Option<usize>,
    pub rope_params: Option<RopeParameters>,
}

impl Ministral3Model {
    /// Forward pass through the entire model
    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        // Find indices for mask creation
        let fa_idx = self.layers.iter().position(|l| !l.use_sliding).unwrap_or(0);
        let swa_idx = self.layers.iter().position(|l| l.use_sliding);

        // Get offsets
        let fa_offset = caches[fa_idx].as_interface().offset();
        let swa_offset = swa_idx
            .map(|idx| caches[idx].as_interface().offset())
            .unwrap_or(0);

        // Embed tokens first so seq_len is derived from h (matching upstream mlx-vlm fix:
        // use h.shape[1] instead of inputs.shape[1] for attn_scale computation, since
        // h may differ from inputs when inputs_embeds are provided in VLM contexts)
        let mut h = self.embed_tokens.forward(input_ids);

        let h_shape = mlxcel_core::array_shape(&h);
        let seq_len = h_shape[1];

        // Create masks
        let fa_mask = Some(create_causal_mask(seq_len, fa_offset));
        let swa_mask = swa_idx.map(|_| {
            // Full-width windowed mask for a fresh single-pass prefill that
            // exceeds the window (RotatingKVCache keeps all prefill keys),
            // clamped mask otherwise. See issue #408.
            let max_cache = self.sliding_window.map(|w| w as i32).unwrap_or(i32::MAX);
            create_sliding_window_prefill_mask(seq_len, swa_offset, max_cache)
        });

        // Compute Llama 4 attention scale using h.shape[1] (after embedding)
        let attn_scale = if let Some(ref params) = self.rope_params {
            get_llama4_attn_scale(
                seq_len,
                fa_offset,
                params.llama_4_scaling_beta,
                params.original_max_position_embeddings,
            )
        } else {
            vec![1.0; seq_len as usize]
        };

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.use_sliding {
                swa_mask.as_ref().map(|m| m.as_ref().unwrap())
            } else {
                fa_mask.as_ref().map(|m| m.as_ref().unwrap())
            };
            h = layer.forward(&h, &attn_scale, caches[i].as_interface(), mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(head) = &self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.use_sliding {
                    Cache::Rotating(RotatingKVCache::new(
                        self.sliding_window.unwrap_or(512) as i32
                    ))
                } else {
                    Cache::Standard(KVCache::new())
                }
            })
            .collect()
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

    /// Load model from VLM wrapper directory (extracts text_config)
    pub fn load_from_text_config<P: AsRef<Path>>(
        model_dir: P,
    ) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config and extract text_config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let text_config = config
            .get("text_config")
            .ok_or("Missing text_config in VLM wrapper config")?;
        let args: ModelArgs = serde_json::from_value(text_config.clone())
            .map_err(|e| format!("Failed to parse text_config: {}", e))?;

        // Load weights - VLM weights have language_model. prefix
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model with language_model. prefix
        let model = Self::from_weights_with_prefix(&weights, &args, "language_model.")?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        Self::from_weights_with_prefix(weights, args, "")
    }

    /// Create model from loaded weights with optional prefix (for VLM wrappers)
    pub fn from_weights_with_prefix(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}model.embed_tokens", prefix),
            group_size,
            bits,
        )?;

        // Get layer types
        let layer_types = args.layer_types_vec();

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for (i, layer_type) in layer_types.iter().enumerate() {
            let use_sliding = layer_type == "sliding_attention";
            let layer =
                TransformerBlock::from_weights_with_prefix(weights, args, i, use_sliding, prefix)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, &format!("{}model.norm.weight", prefix))?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head (or use tied embeddings)
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}lm_head", prefix),
                group_size,
                bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            rope_params: args.rope_parameters.clone(),
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
use std::cell::RefCell;

/// Wrapper for Ministral3Model that implements LanguageModel trait
/// Uses internal cache management for sliding window attention
pub struct Ministral3Wrapper {
    model: Ministral3Model,
    caches: RefCell<Vec<Cache>>,
}

impl Ministral3Wrapper {
    pub fn new(model: Ministral3Model) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches: RefCell::new(caches),
        }
    }

    pub fn reset_caches(&self) {
        let caches = self.model.make_caches();
        *self.caches.borrow_mut() = caches;
    }
}

impl mlxcel_core::generate::LanguageModel for Ministral3Wrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches(input_ids, &mut caches)
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        // Reset internal caches
        self.reset_caches();
        // Return dummy caches (won't be used - internal caches are used instead)
        (0..self.model.layers.len())
            .map(|_| mlxcel_core::layers::KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn supports_batching(&self) -> bool {
        false // Ministral3 uses internal RefCell mixed caches, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // Mistral/Ministral EOS token
    }
}
