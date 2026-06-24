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

//! OLMo3 model implementation using mlxcel-core
//!
//! OLMo3 = OLMo2 + Sliding Window Attention
//! Key features:
//! - Post-norm architecture (norm AFTER attention/MLP output, not before)
//! - Q/K normalization (separate RMSNorm applied to Q and K before reshape)
//! - Sliding window attention with layer_types config
//! - Different attention patterns: full_attention vs sliding_attention
//! - Standard SwiGLU MLP

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask_dense};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct OLMo3Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub sliding_window: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl OLMo3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
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

    /// Get layer types, defaulting to pattern where every 4th layer is full attention
    pub fn get_layer_types(&self) -> Vec<String> {
        self.layer_types.clone().unwrap_or_else(|| {
            // Default: every 4th layer is full_attention, others are sliding_attention
            (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect()
        })
    }

    /// Returns true if the layer uses sliding window attention
    pub fn is_sliding_window_layer(&self, layer_idx: usize) -> bool {
        let layer_types = self.get_layer_types();
        layer_types[layer_idx] != "full_attention"
    }
}

// OLMo3 Attention (with Q/K norm and optional sliding window).
pub struct OLMo3Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    // OLMo3 specific: Q/K normalization
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub is_sliding: bool,
    /// Sliding-window size for sliding layers (0 for full-attention layers).
    /// Used to slice K/V to match the post-earlier capped sliding mask.
    pub window_size: i32,
}

impl OLMo3Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // OLMo3 specific: Apply Q/K norms BEFORE reshape
        // Norms operate on full dimension [B, L, n_heads * head_dim]
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_some() {
            // Prefill: use mask. A sliding-window layer's mask is either the
            // clamped `(l, window_size)` mask or the full `(l, k_len)` mask for
            // a fresh single-pass prefill that exceeds the window (issue #408).
            // Plain `KVCache` returns full-length K/V, so slice K/V to the
            // mask's key axis (not blindly to `window_size`): a full mask keeps
            // every key, a clamped mask drops the oldest. Mirrors
            // `causal_attention`'s internal handling.
            let k_shape = mlxcel_core::array_shape(&cache_k);
            let k_len = k_shape[2];
            let mask_klen = mask
                .map(|m| *mlxcel_core::array_shape(m).last().unwrap_or(&k_len))
                .unwrap_or(k_len);
            let (k_used, v_used) = if self.window_size > 0 && k_len > mask_klen {
                let v_shape = mlxcel_core::array_shape(&cache_v);
                let start = k_len - mask_klen;
                (
                    Some(mlxcel_core::slice(
                        &cache_k,
                        &[0, 0, start, 0],
                        &[k_shape[0], k_shape[1], k_len, k_shape[3]],
                    )),
                    Some(mlxcel_core::slice(
                        &cache_v,
                        &[0, 0, start, 0],
                        &[v_shape[0], v_shape[1], k_len, v_shape[3]],
                    )),
                )
            } else {
                (None, None)
            };
            let k_ref: &MlxArray = k_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_k.as_ref().unwrap());
            let v_ref: &MlxArray = v_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_v.as_ref().unwrap());

            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, k_ref, v_ref, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else if l > 1 {
            // Prefill without mask: use causal
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            // Single token: use causal SDPA
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &OLMo3Config,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let is_sliding = args.is_sliding_window_layer(layer_idx);

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Load Q/K norms
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = RMSNorm::new(q_norm_weight, args.rms_norm_eps);
        let k_norm = RMSNorm::new(k_norm_weight, args.rms_norm_eps);

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            is_sliding,
            window_size: if is_sliding {
                args.sliding_window as i32
            } else {
                0
            },
        })
    }
}

// MLP (SwiGLU).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // Use compiled SwiGLU for kernel fusion
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &OLMo3Config,
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

// OLMo3 Transformer Block (Post-norm architecture).
pub struct OLMo3TransformerBlock {
    pub self_attn: OLMo3Attention,
    pub mlp: MLP,
    // OLMo3 uses POST-norm: norm AFTER attention/MLP output
    pub post_attention_layernorm: RMSNorm,
    pub post_feedforward_layernorm: RMSNorm,
}

impl OLMo3TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // OLMo3 POST-norm: norm AFTER the operation, then add residual
        // h = x + post_attention_layernorm(self_attn(x))
        let attn_out = self.self_attn.forward(x, cache, mask);
        let normed_attn = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::add(x, &normed_attn);

        // out = h + post_feedforward_layernorm(mlp(h))
        let mlp_out = self.mlp.forward(&h);
        let normed_mlp = self.post_feedforward_layernorm.forward(&mlp_out);
        mlxcel_core::add(&h, &normed_mlp)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &OLMo3Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = OLMo3Attention::from_weights(
            weights,
            args,
            layer_idx,
            &format!("{}.self_attn", prefix),
        )?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        // POST-norm weights
        let post_attn_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_ff_weight = get_weight_copy(
            weights,
            &format!("{}.post_feedforward_layernorm.weight", prefix),
        )?;

        let post_attention_layernorm = RMSNorm::new(post_attn_weight, args.rms_norm_eps);
        let post_feedforward_layernorm = RMSNorm::new(post_ff_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            post_attention_layernorm,
            post_feedforward_layernorm,
        })
    }
}

// OLMo3 Model.
pub struct OLMo3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<OLMo3TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub config: OLMo3Config,
    // Pre-computed layer type info
    layer_types: Vec<String>,
    swa_idx: usize,
    ga_idx: usize,
}

impl OLMo3Model {
    /// Forward pass through the entire model
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1] as usize;

        // Create masks for full and sliding attention
        let (full_mask, sliding_mask) = if l > 1 {
            // Size the prefill masks from the cache's live window
            // (`live_len() = offset - live_start`), not the monotonic
            // `offset`. Under `--max-kv-size`, `trim_front` slices the buffer
            // to the live window and advances `live_start` while `offset`
            // keeps growing to preserve the RoPE relative positions, so
            // `update_and_fetch` returns only `live_len` keys. A mask sized
            // from `offset` would be wider than the returned K/V and break the
            // attention broadcast. With no trim (`live_start == 0`),
            // `live_len == offset`, so this is byte-identical to the untrimmed
            // path. See issue #419.
            let ga_live_len = caches[self.ga_idx].live_len();
            let swa_live_len = caches[self.swa_idx].live_len();

            let full = Some(create_causal_mask(l as i32, ga_live_len));
            // Dense `KVCache` keeps every key in the live window, so the
            // prefill mask is the full windowed-causal mask over the retained
            // (live) keys; the attention layer slices K/V to the mask's key
            // axis. The window is enforced by the mask, not by dropping keys.
            // See issues #408, #413, #419.
            let sliding = Some(create_sliding_window_prefill_mask_dense(
                l as i32,
                swa_live_len,
                self.config.sliding_window as i32,
            ));
            (full, sliding)
        } else {
            (None, None)
        };

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.layer_types[i] == "full_attention" {
                full_mask.as_ref().map(|m| m.as_ref().unwrap())
            } else {
                sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
            };
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Create KV caches for all layers
    pub fn make_caches_impl(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from directory
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, OLMo3Config), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: OLMo3Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &OLMo3Config) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let layer_types = args.get_layer_types();

        // Find first indices for each attention type
        let swa_idx = layer_types
            .iter()
            .position(|t| t == "sliding_attention")
            .unwrap_or(0);
        let ga_idx = layer_types
            .iter()
            .position(|t| t == "full_attention")
            .unwrap_or(0);

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = OLMo3TransformerBlock::from_weights(weights, args, i)?;
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
            config: args.clone(),
            layer_types,
            swa_idx,
            ga_idx,
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
impl LanguageModel for OLMo3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_caches_impl()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // OLMo3 uses standard EOS token
        vec![100257] // <|endoftext|>
    }
}
