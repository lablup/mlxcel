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

//! BitNet b1.58 (ternary-weight) model.
//!
//! Llama-style transformer where every projection is a `BitLinear`: 1.58-bit
//! ternary weights ({-1, 0, +1}) packed 4-per-uint8, multiplied directly on the
//! packed bytes by a custom Metal kernel (`bitlinear_matmul`) scaled by a single
//! `weight_scale`. Two extra differences from Llama: an `attn_sub_norm` before
//! `o_proj`, an `ffn_sub_norm` inside the MLP, and a squared-ReLU MLP
//! (`relu2(gate) * up`). Embedding and (untied) lm_head stay full precision.
//!
//! Reference:
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bitnet.py
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bitlinear_layers.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct BitNetConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub eos_token_id: Option<i32>,
    #[serde(default)]
    pub quantization_config: Option<BitNetQuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BitNetQuantConfig {
    #[serde(default)]
    pub linear_class: Option<String>,
}

fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_true() -> bool {
    true
}

impl BitNetConfig {
    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads) as i32
    }

    /// Scales are inverted unless the checkpoint uses the `autobitlinear`
    /// linear class (mlx-lm `bitnet_quantize`).
    pub fn invert_weight_scales(&self) -> bool {
        self.quantization_config
            .as_ref()
            .and_then(|q| q.linear_class.as_deref())
            != Some("autobitlinear")
    }
}

// BitLinear: ternary-weight linear with a single output scale.
pub struct BitLinear {
    packed_weight: UniquePtr<MlxArray>, // [out/4, in] uint8
    weight_scale: UniquePtr<MlxArray>,  // [1]
    bias: Option<UniquePtr<MlxArray>>,
    in_features: i32,
    out_features: i32,
    invert_weight_scales: bool,
}

impl BitLinear {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_features: i32,
        out_features: i32,
        invert_weight_scales: bool,
    ) -> Result<Self, String> {
        let packed_weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let weight_scale = get_weight_copy(weights, &format!("{}.weight_scale", prefix))?;
        let bias = weights
            .get(&format!("{}.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        Ok(Self {
            packed_weight,
            weight_scale,
            bias,
            in_features,
            out_features,
            invert_weight_scales,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::bitlinear_matmul(
            x,
            &self.packed_weight,
            &self.weight_scale,
            self.in_features,
            self.out_features,
            self.invert_weight_scales,
        );
        match &self.bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        }
    }
}

// Attention.
pub struct BitNetAttention {
    q_proj: BitLinear,
    k_proj: BitLinear,
    v_proj: BitLinear,
    o_proj: BitLinear,
    attn_sub_norm: RMSNorm,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_base: f32,
    rope_traditional: bool,
}

impl BitNetAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BitNetConfig,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size as i32;
        let n_heads = cfg.num_attention_heads as i32;
        let n_kv_heads = cfg.num_key_value_heads as i32;
        let head_dim = cfg.head_dim();
        let invert = cfg.invert_weight_scales();

        let q_proj = BitLinear::from_weights(
            weights,
            &format!("{}.q_proj", prefix),
            hidden,
            n_heads * head_dim,
            invert,
        )?;
        let k_proj = BitLinear::from_weights(
            weights,
            &format!("{}.k_proj", prefix),
            hidden,
            n_kv_heads * head_dim,
            invert,
        )?;
        let v_proj = BitLinear::from_weights(
            weights,
            &format!("{}.v_proj", prefix),
            hidden,
            n_kv_heads * head_dim,
            invert,
        )?;
        let o_proj = BitLinear::from_weights(
            weights,
            &format!("{}.o_proj", prefix),
            n_heads * head_dim,
            hidden,
            invert,
        )?;
        let attn_sub_norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.attn_sub_norm.weight", prefix))?,
            cfg.rms_norm_eps,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn_sub_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_base: cfg.rope_theta,
            rope_traditional: cfg.rope_traditional,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(
            &q,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

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

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);

        // BitNet: sub-norm on the attention output before o_proj.
        let attn_out = self.attn_sub_norm.forward(&attn_out);
        self.o_proj.forward(&attn_out)
    }
}

// MLP: relu2(gate) * up, sub-norm, then down.
pub struct BitNetMLP {
    gate_proj: BitLinear,
    up_proj: BitLinear,
    down_proj: BitLinear,
    ffn_sub_norm: RMSNorm,
}

impl BitNetMLP {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BitNetConfig,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size as i32;
        let inter = cfg.intermediate_size as i32;
        let invert = cfg.invert_weight_scales();

        Ok(Self {
            gate_proj: BitLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                hidden,
                inter,
                invert,
            )?,
            up_proj: BitLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                hidden,
                inter,
                invert,
            )?,
            down_proj: BitLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                inter,
                hidden,
                invert,
            )?,
            ffn_sub_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.ffn_sub_norm.weight", prefix))?,
                cfg.rms_norm_eps,
            ),
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::multiply(&mlxcel_core::compiled_relu_squared(&gate), &up);
        let normed = self.ffn_sub_norm.forward(&activated);
        self.down_proj.forward(&normed)
    }
}

// Decoder layer.
pub struct BitNetDecoderLayer {
    self_attn: BitNetAttention,
    mlp: BitNetMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl BitNetDecoderLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BitNetConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: BitNetAttention::from_weights(
                weights,
                &format!("{}.self_attn", prefix),
                cfg,
            )?,
            mlp: BitNetMLP::from_weights(weights, &format!("{}.mlp", prefix), cfg)?,
            input_layernorm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
                cfg.rms_norm_eps,
            ),
            post_attention_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                cfg.rms_norm_eps,
            ),
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Model.
pub struct BitNetModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<BitNetDecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    eos_token_ids: Vec<i32>,
}

impl BitNetModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, BitNetConfig), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: BitNetConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &config)?;
        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &BitNetConfig) -> Result<Self, String> {
        // Embedding / lm_head are full precision (not BitLinear); group_size/bits
        // only matter for the affine-quantized -4/6/8bit variants.
        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", 64, 4)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(BitNetDecoderLayer::from_weights(
                weights,
                &format!("model.layers.{}", i),
                config,
            )?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            config.rms_norm_eps,
        );

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(weights, "lm_head", 64, 4)?)
        };

        let eos_token_ids = config.eos_token_id.map(|e| vec![e]).unwrap_or_default();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            eos_token_ids,
        })
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

impl LanguageModel for BitNetModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        BitNetModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        BitNetModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    /// Known ternary case for the packed BitLinear matmul.
    ///
    /// W (out=4, in=4), packed 4 output rows per uint8 column (value = bits - 1):
    ///   row0 = [ 1,  0, -1,  0]
    ///   row1 = [ 0,  1,  0, -1]
    ///   row2 = [-1, -1,  1,  1]
    ///   row3 = [ 1,  1,  0,  0]
    /// packed[0, col] bits {row0,row1,row2,row3} -> bytes [134, 137, 100, 97].
    /// x = [1, 2, 3, 4], scale = 2.0 (not inverted) =>
    ///   y = (x @ W^T) * 2 = [-4, -4, 8, 6].
    #[test]
    fn bitlinear_matmul_known_ternary_case() {
        let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let packed =
            mlxcel_core::from_bytes(&[134u8, 137, 100, 97], &[1, 4], mlxcel_core::dtype::UINT8);
        let scale = mlxcel_core::from_slice_f32(&[2.0], &[1]);
        let y = mlxcel_core::bitlinear_matmul(&x, &packed, &scale, 4, 4, false);
        let expected = mlxcel_core::from_slice_f32(&[-4.0, -4.0, 8.0, 6.0], &[1, 4]);
        let diff = mlxcel_core::abs(&mlxcel_core::subtract(&y, &expected));
        let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_axis(
            &mlxcel_core::reshape(&diff, &[-1]),
            -1,
            false,
        ));
        assert!(
            max_abs < 1e-4,
            "BitLinear output mismatch, max_abs = {max_abs}"
        );
    }
}
