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

//! Baichuan model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Fused W_pack QKV projection (Q, K, V concatenated in one weight)
//! - Sliding window attention with separate head configs for SWA layers
//! - RMSNorm + SiLU (standard Llama style)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;
use std::path::Path;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct BaichuanConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub sliding_window_layers: Vec<usize>,
    pub conv_window: usize,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub num_swa_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_swa_key_value_heads: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl BaichuanConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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

// Attention.
pub struct BaichuanAttention {
    pub w_pack: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub is_swa: bool,
    pub sliding_window: Option<i32>,
    // Custom convolution weights for K and V
    pub conv_k: UniquePtr<MlxArray>,
    pub conv_v: UniquePtr<MlxArray>,
    pub conv_window: i32,
}

impl BaichuanAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BaichuanConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let is_swa = cfg.sliding_window_layers.contains(&layer_idx);
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        let n_heads = if is_swa {
            cfg.num_swa_attention_heads
                .unwrap_or(cfg.num_attention_heads)
        } else {
            cfg.num_attention_heads
        } as i32;

        let n_kv_heads = if is_swa {
            cfg.num_swa_key_value_heads
                .unwrap_or(cfg.num_key_value_heads)
        } else {
            cfg.num_key_value_heads
        } as i32;

        // head_dim is computed based on the number of heads for this layer type
        // SWA layers may have different head counts, so head_dim differs
        let head_dim = (cfg.hidden_size / n_heads as usize) as i32;

        let sliding_window = if is_swa {
            Some(cfg.sliding_window as i32)
        } else {
            None
        };

        // Load convolution weights for K and V
        let conv_k = get_weight_copy(weights, &format!("{}.conv_k", prefix))?;
        let conv_v = get_weight_copy(weights, &format!("{}.conv_v", prefix))?;

        Ok(Self {
            w_pack: UnifiedLinear::from_weights(
                weights,
                &format!("{}.W_pack", prefix),
                group_size,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.o_proj", prefix),
                group_size,
                bits,
            )?,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_base: cfg.rope_theta,
            is_swa,
            sliding_window,
            conv_k,
            conv_v,
            conv_window: cfg.conv_window as i32,
        })
    }

    /// Custom 1D convolution for K/V preprocessing
    /// Input u has shape [B, H, L, D], weights shape [1, 1, H, 1, 2]
    /// Returns u_prev * w0 + u * w1 where w0, w1 are the two conv weights
    fn custom_convolution(
        &self,
        u: &MlxArray,
        weights: &MlxArray,
        state: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(u);
        let b = shape[0];
        let h = shape[1];
        let l = shape[2];
        let d = shape[3];

        // Reshape weights from [1, 1, H, 1, 2] to [1, H, 2, 1, 1]
        // Then slice to get w0 and w1
        let weights_reshaped = mlxcel_core::reshape(weights, &[1, h, self.conv_window, 1, 1]);
        let w0 = mlxcel_core::utils::slice_axis(&weights_reshaped, 2, 0, 1); // [1, H, 1, 1, 1]
        let w1 = mlxcel_core::utils::slice_axis(&weights_reshaped, 2, 1, 2); // [1, H, 1, 1, 1]

        // Squeeze axis 2 to get [1, H, 1, 1]
        let w0 = mlxcel_core::squeeze_axis(&w0, 2);
        let w1 = mlxcel_core::squeeze_axis(&w1, 2);

        // Create u_prev: concatenate state with u[:, :, :-1]
        let u_dtype = mlxcel_core::array_dtype(u);
        let u_prev = if l > 1 {
            // For prefill: concat zeros (or state) with u[:-1]
            let zero_state = if let Some(s) = state {
                mlxcel_core::copy(s)
            } else {
                mlxcel_core::zeros(&[b, h, 1, d], u_dtype)
            };
            let u_slice = mlxcel_core::utils::slice_axis(u, 2, 0, l - 1);
            mlxcel_core::concatenate(&zero_state, &u_slice, 2)
        } else {
            // For single token decode: use state directly
            if let Some(s) = state {
                mlxcel_core::copy(s)
            } else {
                mlxcel_core::zeros(&[b, h, 1, d], u_dtype)
            }
        };

        // Compute u_prev * w0 + u * w1
        let term1 = mlxcel_core::multiply(&u_prev, &w0);
        let term2 = mlxcel_core::multiply(u, &w1);
        mlxcel_core::add(&term1, &term2)
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        conv_state: &mut Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let dim = shape[2];

        // Fused QKV projection
        let qkv = self.w_pack.forward(x);

        // Split into Q, K, V
        // W_pack = [Q (dim) | K (n_kv_heads * head_dim) | V (n_kv_heads * head_dim)]
        let kv_size = self.n_kv_heads * self.head_dim;

        let q = mlxcel_core::slice_last_dim(&qkv, 0, dim);
        let k = mlxcel_core::slice_last_dim(&qkv, dim, dim + kv_size);
        let v = mlxcel_core::slice_last_dim(&qkv, dim + kv_size, dim + 2 * kv_size);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Store k_init, v_init for conv state update (last token before conv)
        let k_last = mlxcel_core::utils::slice_axis(&k, 2, l - 1, l);
        let v_last = mlxcel_core::utils::slice_axis(&v, 2, l - 1, l);

        // Apply custom convolution to K and V
        // UniquePtr::as_ref() returns Option<&T>, so we need to flatten
        let last_k_state: Option<&MlxArray> = conv_state.as_ref().and_then(|(k_s, _)| k_s.as_ref());
        let last_v_state: Option<&MlxArray> = conv_state.as_ref().and_then(|(_, v_s)| v_s.as_ref());
        let k = self.custom_convolution(&k, &self.conv_k, last_k_state);
        let v = self.custom_convolution(&v, &self.conv_v, last_v_state);

        // Update conv state with last k/v before convolution
        *conv_state = Some((k_last, v_last));

        let offset = cache.offset;

        // Apply RoPE using fast_rope directly (after convolution)
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(
                &q,
                &cache_k,
                &cache_v,
                self.scale,
                0.0,
                self.sliding_window.unwrap_or(0),
            )
        } else {
            // Single token or explicit mask
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.sliding_window.unwrap_or(0),
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// MLP.
pub struct BaichuanMLP {
    pub gate_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
}

impl BaichuanMLP {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BaichuanConfig,
    ) -> Result<Self, String> {
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // Use compiled SwiGLU activation for kernel fusion
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }
}

// Transformer Block.
pub struct BaichuanDecoderLayer {
    pub self_attn: BaichuanAttention,
    pub mlp: BaichuanMLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl BaichuanDecoderLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BaichuanConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let self_attn = BaichuanAttention::from_weights(
            weights,
            &format!("{}.self_attn", prefix),
            cfg,
            layer_idx,
        )?;
        let mlp = BaichuanMLP::from_weights(weights, &format!("{}.mlp", prefix), cfg)?;

        // Load RMSNorm weights manually
        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, cfg.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, cfg.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        conv_state: &mut Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Attention block with residual
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, conv_state, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // MLP block with residual
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Model.
/// Convolution state for Baichuan attention (stores last K/V before conv)
pub type ConvState = Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>;

pub struct BaichuanModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<BaichuanDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub conv_states: RefCell<Vec<ConvState>>,
}

impl BaichuanModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        let mut conv_states = self.conv_states.borrow_mut();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], &mut conv_states[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn reset_conv_states(&self) {
        let mut conv_states = self.conv_states.borrow_mut();
        for state in conv_states.iter_mut() {
            *state = None;
        }
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, BaichuanConfig), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: BaichuanConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &BaichuanConfig) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Load embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            layers.push(BaichuanDecoderLayer::from_weights(
                weights, &prefix, config, i,
            )?);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        // Load lm_head (if not tied)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        // Initialize conv states (one per layer)
        let conv_states = RefCell::new((0..config.num_hidden_layers).map(|_| None).collect());

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            conv_states,
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
impl LanguageModel for BaichuanModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        BaichuanModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        BaichuanModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Baichuan EOS token
        vec![2] // </s>
    }
}
