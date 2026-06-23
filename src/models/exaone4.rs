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

//! ExaOne 4 model implementation using mlxcel-core
//!
//! Key features:
//! - Sliding window pattern string: "L" for local, "G" for global
//! - Local layers: sliding window mask + RoPE + RotatingKVCache
//! - Global layers: full attention + NO RoPE + standard KVCache
//! - Q/K RMSNorm normalization
//! - Output-norm architecture: h = x + norm(attn_out), out = h + norm(mlp_out)

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
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    pub rope_scaling: std::collections::HashMap<String, serde_json::Value>,

    #[serde(default)]
    pub sliding_window: Option<usize>,

    #[serde(default)]
    pub sliding_window_pattern: Option<String>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

impl ModelArgs {
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
    pub is_local: bool,
    pub window_size: i32,
    pub use_rope: bool,
    pub rope_base: f32,
    /// Pre-computed frequencies for llama3/yarn RoPE scaling (None = use base theta)
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
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
        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Apply Q/K normalization BEFORE transpose
        q = self.q_norm.forward(&q);
        k = self.k_norm.forward(&k);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();

        // Apply RoPE (local layers always, global layers only when no sliding window pattern)
        let (q, k) = if self.use_rope {
            if let Some(ref freqs) = self.rope_freqs {
                // Llama3/Yarn RoPE with precomputed frequencies
                let q_rope =
                    mlxcel_core::fast_rope_with_freqs(&q, self.head_dim, false, 1.0, offset, freqs);
                let k_rope =
                    mlxcel_core::fast_rope_with_freqs(&k, self.head_dim, false, 1.0, offset, freqs);
                (q_rope, k_rope)
            } else {
                let q_rope =
                    mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
                let k_rope =
                    mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);
                (q_rope, k_rope)
            }
        } else {
            (q, k)
        };

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
        is_local: Option<bool>,
        rope_freqs: Option<UniquePtr<MlxArray>>,
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

        let head_dim = args.head_dim as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Load Q/K normalization
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = RMSNorm::new(q_norm_weight, args.rms_norm_eps);
        let k_norm = RMSNorm::new(k_norm_weight, args.rms_norm_eps);

        // Python: self.use_rope = is_local is None or is_local
        // None (no pattern) → all layers use RoPE
        // Some(true) → local layers use RoPE
        // Some(false) → global layers don't use RoPE
        let use_rope = is_local.unwrap_or(true);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale,
            is_local: is_local.unwrap_or(false),
            window_size: if is_local.unwrap_or(false) {
                args.sliding_window.unwrap_or(512) as i32
            } else {
                0
            },
            use_rope,
            rope_base: args.rope_theta,
            rope_freqs,
            q_norm,
            k_norm,
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
        if let Some(result) = mlxcel_core::layers::compiled_swiglu_mlp(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

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

// Transformer Block (Output-norm: norm outputs before residual add).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub post_attention_layernorm: RMSNorm,
    pub post_feedforward_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Attention: norm output THEN add residual
        // Python: h = x + self.post_attention_layernorm(r)
        let attn_out = self.self_attn.forward(x, cache, mask);
        let attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::add(x, &attn_normed);

        // MLP: norm output THEN add residual
        // Python: out = h + self.post_feedforward_layernorm(r)
        let mlp_out = self.mlp.forward(&h);
        let mlp_normed = self.post_feedforward_layernorm.forward(&mlp_out);
        mlxcel_core::add(&h, &mlp_normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        is_local: Option<bool>,
        rope_freqs: Option<UniquePtr<MlxArray>>,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(
            weights,
            args,
            &format!("{}.self_attn", prefix),
            is_local,
            rope_freqs,
        )?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_feedforward_layernorm.weight", prefix),
        )?;

        let post_attention_layernorm = RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);
        let post_feedforward_layernorm = RMSNorm::new(post_ff_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            post_attention_layernorm,
            post_feedforward_layernorm,
        })
    }
}

// Cache Interface (trait for KVCache / RotatingKVCache polymorphism).
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

// ExaOne4 Model.
pub struct ExaOne4Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub sliding_window: Option<usize>,
    pub pattern: Option<Vec<char>>,
}

impl ExaOne4Model {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        mask_local: Option<&MlxArray>,
        mask_global: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.self_attn.is_local {
                mask_local
            } else {
                mask_global
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
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

    /// Create KV caches for all layers (local = RotatingKVCache, global = KVCache)
    pub fn make_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.self_attn.is_local {
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

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Parse sliding window pattern
        let pattern = args
            .sliding_window_pattern
            .as_ref()
            .map(|s| s.chars().collect::<Vec<_>>());

        // Compute RoPE frequencies (llama3 scaling if configured)
        let rope_freqs = compute_rope_freqs(args);

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            // Python: is_local = pattern[i % len(pattern)] == "L" if pattern else None
            let is_local = pattern.as_ref().map(|pat| pat[i % pat.len()] == 'L');

            // Clone rope_freqs for layers that use RoPE
            let use_rope = is_local.unwrap_or(true);
            let layer_freqs = if use_rope {
                rope_freqs.as_ref().map(|f| mlxcel_core::copy(f))
            } else {
                None
            };

            let layer = TransformerBlock::from_weights(weights, args, i, is_local, layer_freqs)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head (or use tied embeddings)
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            pattern: pattern.clone(),
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

/// Compute llama3 RoPE frequencies from rope_scaling config.
/// Returns None for default RoPE (no scaling or unsupported type).
fn compute_rope_freqs(args: &ModelArgs) -> Option<UniquePtr<MlxArray>> {
    let scaling = &args.rope_scaling;
    let rope_type = scaling
        .get("rope_type")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    if rope_type != "llama3" {
        return None;
    }

    let factor = scaling
        .get("factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let low_freq_factor = scaling
        .get("low_freq_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let high_freq_factor = scaling
        .get("high_freq_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(4.0) as f32;
    let old_context_len = scaling
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_f64())
        .unwrap_or(8192.0) as f32;

    let dims = args.head_dim;
    let base = args.rope_theta;

    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;

    // Compute freqs = base^(arange(0, dims, 2) / dims)
    let half_dims = dims / 2;
    let mut freq_vals = Vec::with_capacity(half_dims);
    for i in 0..half_dims {
        let exp = (2 * i) as f32 / dims as f32;
        let freq = base.powf(exp);
        let wavelen = 2.0 * std::f32::consts::PI * freq;

        let adjusted = if wavelen > low_freq_wavelen {
            // Low frequency: scale by factor
            freq * factor
        } else if wavelen > high_freq_wavelen {
            // Medium frequency: smooth interpolation
            let smooth = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            freq / ((1.0 - smooth) / factor + smooth)
        } else {
            // High frequency: no change
            freq
        };
        freq_vals.push(adjusted);
    }

    Some(mlxcel_core::from_slice_f32(&freq_vals, &[half_dims as i32]))
}

// LanguageModel trait implementation.
use std::cell::RefCell;

impl ExaOne4Model {
    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[1];

        // Get offsets for mask creation
        let local_idx = self
            .layers
            .iter()
            .position(|l| l.self_attn.is_local)
            .unwrap_or(0);
        let global_idx = self
            .layers
            .iter()
            .position(|l| !l.self_attn.is_local)
            .unwrap_or(0);

        let local_offset = caches[local_idx].as_interface().offset();
        let global_offset = caches[global_idx].as_interface().offset();

        // Create masks (cast to bfloat16 to match model weights dtype for SDPA)
        let mask_local = if self.sliding_window.is_some() {
            let max_cache = self.sliding_window.map(|w| w as i32).unwrap_or(i32::MAX);
            // Full-width windowed mask for a fresh single-pass prefill that
            // exceeds the window (RotatingKVCache keeps all prefill keys),
            // clamped mask otherwise. See issue #408.
            let mask = create_sliding_window_prefill_mask(seq_len, local_offset, max_cache);
            Some(mlxcel_core::astype(&mask, mlxcel_core::dtype::BFLOAT16))
        } else {
            None
        };

        let mask_global = Some(mlxcel_core::astype(
            &create_causal_mask(seq_len, global_offset),
            mlxcel_core::dtype::BFLOAT16,
        ));

        self.forward(
            input_ids,
            caches,
            mask_local.as_ref().map(|m| m.as_ref().unwrap()),
            mask_global.as_ref().map(|m| m.as_ref().unwrap()),
        )
    }
}

/// Wrapper for ExaOne4Model that implements LanguageModel trait
/// Uses internal cache management for sliding window attention
pub struct ExaOne4Wrapper {
    model: ExaOne4Model,
    caches: RefCell<Vec<Cache>>,
}

impl ExaOne4Wrapper {
    pub fn new(model: ExaOne4Model) -> Self {
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

impl mlxcel_core::generate::LanguageModel for ExaOne4Wrapper {
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
        false // ExaOne4 uses internal RefCell mixed caches, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![361] // ExaOne4 EOS token: [|endofturn|]
    }
}
