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

// Nemotron-NAS: NVIDIA's Neural Architecture Search model for mlxcel-core
// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron-nas.py
//
// Key features:
// - Per-layer configuration via block_configs (AttentionConfig, FFNConfig)
// - Heterogeneous transformer with variable GQA and FFN sizes
// - Optional no-op or linear replacement for attention/MLP
// - Standard GQA attention with RoPE
// - AttentionSubblock enum: None, Linear, Attention
// - FFNSubblock enum: None, Linear, MLP
// - ffn_mult_to_intermediate_size for variable FFN widths

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, repeat_kv};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{
    MlxArray, UniquePtr, add, array_shape, copy, fast_rope, gelu, multiply, relu, reshape, silu,
    transpose_axes,
};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(alias = "type")]
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AttentionConfig {
    #[serde(default)]
    pub no_op: bool,
    #[serde(default)]
    pub replace_with_linear: bool,
    #[serde(default)]
    pub sparsify: Option<Vec<String>>,
    #[serde(default)]
    pub n_heads_in_group: Option<usize>,
    #[serde(default)]
    pub window_length: Option<usize>,
    #[serde(default)]
    pub num_sink_tokens: Option<usize>,
    #[serde(default)]
    pub use_prefill_window_in_sink_attention: bool,
    #[serde(default)]
    pub unshifted_sink: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FFNConfig {
    #[serde(default)]
    pub no_op: bool,
    #[serde(default)]
    pub replace_with_linear: bool,
    #[serde(default)]
    pub sparsify: Option<Vec<String>>,
    #[serde(default)]
    pub ffn_mult: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlockConfig {
    #[serde(default)]
    pub attention: AttentionConfig,
    #[serde(default)]
    pub ffn: FFNConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NemotronNASConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    pub vocab_size: usize,

    pub block_configs: Vec<BlockConfig>,

    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "nemotron-nas".to_string()
}

fn default_hidden_size() -> usize {
    8192
}

fn default_num_attention_heads() -> usize {
    64
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_rope_theta() -> f32 {
    500000.0
}

fn default_max_position_embeddings() -> usize {
    131072
}

impl NemotronNASConfig {
    pub fn get_head_dim(&self) -> usize {
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

// Helper to calculate intermediate size from ffn_mult
fn ffn_mult_to_intermediate_size(ffn_mult: f32, n_embd: usize) -> usize {
    let intermediate_size = (2.0 * ffn_mult * n_embd as f32 / 3.0) as usize;
    // Round up to multiple of 256
    intermediate_size.div_ceil(256) * 256
}

// NAS Attention - Standard GQA with RoPE.
struct NASAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    scale: f32,
}

impl NASAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = reshape(
            &q,
            &[batch, seq_len, self.n_heads as i32, self.head_dim as i32],
        );
        let k = reshape(
            &k,
            &[batch, seq_len, self.n_kv_heads as i32, self.head_dim as i32],
        );
        let v = reshape(
            &v,
            &[batch, seq_len, self.n_kv_heads as i32, self.head_dim as i32],
        );

        // Apply RoPE
        let offset = cache.offset;
        let q = fast_rope(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );
        let k = fast_rope(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = transpose_axes(&q, &[0, 2, 1, 3]);
        let k = transpose_axes(&k, &[0, 2, 1, 3]);
        let v = transpose_axes(&v, &[0, 2, 1, 3]);

        // Update KV cache and get full keys/values
        let (k, v) = cache.update_and_fetch(k, v);

        // Repeat KV for GQA if needed
        let n_rep = (self.n_heads / self.n_kv_heads) as i32;
        let (k, v) = if n_rep > 1 {
            (repeat_kv(&k, n_rep), repeat_kv(&v, n_rep))
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
        };

        // Transpose back and reshape
        let output = transpose_axes(&output, &[0, 2, 1, 3]);
        let output = reshape(
            &output,
            &[batch, seq_len, (self.n_heads * self.head_dim) as i32],
        );

        self.o_proj.forward(&output)
    }
}

// NAS MLP - Gated MLP with configurable activation.
struct NASMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    act_fn: String,
}

impl NASMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // Apply activation function
        let activated = match self.act_fn.as_str() {
            "silu" => silu(&gate),
            "relu" => relu(&gate),
            "gelu" => gelu(&gate),
            _ => silu(&gate),
        };

        let combined = multiply(&activated, &up);
        self.down_proj.forward(&combined)
    }
}

// Linear Replacement for no-op blocks.
struct LinearReplacement {
    linear: UnifiedLinear,
}

impl LinearReplacement {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.linear.forward(x)
    }
}

// Attention and FFN Subblocks.
enum AttentionSubblock {
    None,
    Linear(LinearReplacement),
    Attention(NASAttention),
}

enum FFNSubblock {
    None,
    Linear(LinearReplacement),
    MLP(NASMLP),
}

// NAS Transformer Block with heterogeneous config.
struct NASTransformerBlock {
    input_layernorm: Option<RMSNorm>,
    self_attn: AttentionSubblock,
    post_attention_layernorm: Option<RMSNorm>,
    mlp: FFNSubblock,
}

impl NASTransformerBlock {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut KVCache>,
    ) -> UniquePtr<MlxArray> {
        let mut h = copy(x);

        // Attention part
        match &self.self_attn {
            AttentionSubblock::None => {
                // No-op: skip attention
            }
            AttentionSubblock::Linear(lin) => {
                if let Some(ref norm) = self.input_layernorm {
                    let normed = norm.forward(&h);
                    let attn_out = lin.forward(&normed);
                    h = add(&h, &attn_out);
                }
            }
            AttentionSubblock::Attention(attn) => {
                if let Some(ref norm) = self.input_layernorm
                    && let Some(cache) = cache
                {
                    let normed = norm.forward(&h);
                    let attn_out = attn.forward(&normed, mask, cache);
                    h = add(&h, &attn_out);
                }
            }
        }

        // MLP part
        match &self.mlp {
            FFNSubblock::None => {
                // No-op: skip MLP
            }
            FFNSubblock::Linear(lin) => {
                if let Some(ref norm) = self.post_attention_layernorm {
                    let normed = norm.forward(&h);
                    let mlp_out = lin.forward(&normed);
                    h = add(&h, &mlp_out);
                }
            }
            FFNSubblock::MLP(mlp) => {
                if let Some(ref norm) = self.post_attention_layernorm {
                    let normed = norm.forward(&h);
                    let mlp_out = mlp.forward(&normed);
                    h = add(&h, &mlp_out);
                }
            }
        }

        h
    }

    fn has_attention(&self) -> bool {
        !matches!(self.self_attn, AttentionSubblock::None)
    }
}

// Full NAS Model.
pub struct NemotronNASModel {
    config: NemotronNASConfig,
    embeddings: UnifiedEmbedding,
    layers: Vec<NASTransformerBlock>,
    norm_f: RMSNorm,
    lm_head: Option<UnifiedLinear>,
}

impl NemotronNASModel {
    pub fn load(model_path: &str) -> Result<(Self, NemotronNASConfig), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        println!("[NemotronNAS] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let config: NemotronNASConfig = serde_json::from_str(&config_str)?;

        // Count attention layers for cache sizing
        let num_attention_layers = config
            .block_configs
            .iter()
            .filter(|bc| !bc.attention.no_op)
            .count();

        println!(
            "[NemotronNAS] Config loaded: {} layers ({} with attention, {} no-op)",
            config.num_hidden_layers,
            num_attention_layers,
            config.num_hidden_layers - num_attention_layers
        );

        println!("[NemotronNAS] Loading weights from safetensors...");
        let weights = crate::models::load_text_weights(path, None)?;

        println!("[NemotronNAS] Building model...");
        let model = Self::from_weights(config.clone(), weights)?;

        println!("[NemotronNAS] Model loaded successfully");
        Ok((model, config))
    }

    pub fn from_weights(
        config: NemotronNASConfig,
        mut weights: WeightMap,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Embeddings
        let embeddings =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group_size, bits)?;

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, block_config) in config.block_configs.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);
            let attn_config = &block_config.attention;
            let ffn_config = &block_config.ffn;

            // Input layernorm (needed unless attention is no-op)
            let input_layernorm = if !attn_config.no_op {
                let weight = weights
                    .remove(&format!("{}.input_layernorm.weight", prefix))
                    .ok_or(format!("Missing input_layernorm weight for layer {}", i))?;
                Some(RMSNorm::new(weight, config.rms_norm_eps))
            } else {
                None
            };

            // Attention subblock
            let self_attn = if attn_config.no_op {
                AttentionSubblock::None
            } else if attn_config.replace_with_linear {
                let linear = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.self_attn.linear", prefix),
                    group_size,
                    bits,
                )?;
                AttentionSubblock::Linear(LinearReplacement { linear })
            } else {
                // Full attention
                let n_heads_in_group = attn_config.n_heads_in_group.unwrap_or(1);
                let n_kv_heads = config.num_attention_heads / n_heads_in_group;
                let head_dim = config.get_head_dim();

                let rope_scale = config
                    .rope_scaling
                    .as_ref()
                    .and_then(|s| s.factor)
                    .map(|f| 1.0 / f)
                    .unwrap_or(1.0);

                let q_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.self_attn.q_proj", prefix),
                    group_size,
                    bits,
                )?;
                let k_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.self_attn.k_proj", prefix),
                    group_size,
                    bits,
                )?;
                let v_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.self_attn.v_proj", prefix),
                    group_size,
                    bits,
                )?;
                let o_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.self_attn.o_proj", prefix),
                    group_size,
                    bits,
                )?;

                AttentionSubblock::Attention(NASAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    n_heads: config.num_attention_heads,
                    n_kv_heads,
                    head_dim,
                    rope_dims: head_dim as i32,
                    rope_base: config.rope_theta,
                    rope_scale,
                    scale: (head_dim as f32).powf(-0.5),
                })
            };

            // Post-attention layernorm (needed unless FFN is no-op)
            let post_attention_layernorm = if !ffn_config.no_op {
                let weight = weights
                    .remove(&format!("{}.post_attention_layernorm.weight", prefix))
                    .ok_or(format!(
                        "Missing post_attention_layernorm weight for layer {}",
                        i
                    ))?;
                Some(RMSNorm::new(weight, config.rms_norm_eps))
            } else {
                None
            };

            // FFN subblock
            let mlp = if ffn_config.no_op {
                FFNSubblock::None
            } else if ffn_config.replace_with_linear {
                let linear = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.mlp.linear", prefix),
                    group_size,
                    bits,
                )?;
                FFNSubblock::Linear(LinearReplacement { linear })
            } else {
                // Full MLP
                let ffn_mult = ffn_config.ffn_mult.unwrap_or(4.0);
                let _hidden_dim = ffn_mult_to_intermediate_size(ffn_mult, config.hidden_size);

                let gate_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.mlp.gate_proj", prefix),
                    group_size,
                    bits,
                )?;
                let up_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.mlp.up_proj", prefix),
                    group_size,
                    bits,
                )?;
                let down_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.mlp.down_proj", prefix),
                    group_size,
                    bits,
                )?;

                FFNSubblock::MLP(NASMLP {
                    gate_proj,
                    up_proj,
                    down_proj,
                    act_fn: config.hidden_act.clone(),
                })
            };

            layers.push(NASTransformerBlock {
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                mlp,
            });
        }

        // Final norm
        let norm_f_weight = weights
            .remove("model.norm.weight")
            .ok_or("Missing final norm weight")?;
        let norm_f = RMSNorm::new(norm_f_weight, config.rms_norm_eps);

        // LM head (optional if tied embeddings)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                &weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for NemotronNASModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embeddings.forward(input_ids);

        // Create causal mask if not provided
        let computed_mask = if mask.is_none() {
            let shape = array_shape(&h);
            let seq_len = shape[1];
            let offset = caches.first().map(|c| c.offset).unwrap_or(0);
            if seq_len > 1 {
                Some(create_causal_mask(seq_len, offset))
            } else {
                None
            }
        } else {
            None
        };

        let attn_mask = mask.or(computed_mask.as_deref());

        // Process through layers
        let mut cache_idx = 0;
        for layer in &self.layers {
            if layer.has_attention() {
                h = layer.forward(&h, attn_mask, Some(&mut caches[cache_idx]));
                cache_idx += 1;
            } else {
                h = layer.forward(&h, None, None);
            }
        }

        // Final norm
        let h = self.norm_f.forward(&h);

        // LM head or tied embeddings
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embeddings.as_linear(&h)
        }
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Only create caches for layers with attention
        let num_attention_layers = self
            .layers
            .iter()
            .filter(|layer| layer.has_attention())
            .count();

        (0..num_attention_layers).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt Mamba recurrent state in hybrid architecture
    }

    fn supports_batching(&self) -> bool {
        false // NemotronNAS is a hybrid architecture, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // Standard EOS token for most models
    }
}
