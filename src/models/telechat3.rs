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

//! TeleAI TeleChat3 text model implementation using mlxcel-core.
//!
//! Ported from
//! <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/telechat3.py>.
//!
//! Structurally this is a stock Llama decoder: RMSNorm, GQA, SwiGLU, untied
//! `lm_head`. Two things keep it from simply reusing [`crate::models::llama3`].
//!
//! The first is **YaRN RoPE**. `Telechat3-36B-Thinking` ships
//! `rope_scaling.rope_type = "telechat3-yarn"` with `factor: 4.0`,
//! `original_max_position_embeddings: 8192`, `beta_fast: 32` and
//! `beta_slow: 1`. Upstream routes that straight into the same `YarnRoPE` it
//! uses for `"yarn"` and `"deepseek_yarn"`; the vendor prefix names the
//! checkpoint, not a different algorithm. `llama3::ModelArgs` declares a
//! `rope_scaling` field but never reads it and carries no `beta_fast` /
//! `beta_slow`, so routing TeleChat3 through that path would rotate at the
//! unscaled base. Nothing about that failure is visible on a short prompt: YaRN
//! and default RoPE agree closely at small offsets and only diverge as the
//! position grows past `original_max_position_embeddings`.
//!
//! The second is `attention_bias`, which puts bias terms on q/k/v/o. It is
//! plumbed through because the family declares it, but note that the published
//! 36B checkpoint sets it **false**, so it is not what distinguishes this
//! family in practice.

use crate::models::mellum::compute_yarn_rope;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.

/// Same shape as the per-family enums in [`crate::models::gpt2`] and
/// [`crate::models::helium`]; serde fails the whole config when one field does
/// not match its declared type, so the list form has to be accepted even though
/// published TeleChat3 checkpoints write a single int.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIdField {
    Single(i32),
    Multiple(Vec<i32>),
}

impl TokenIdField {
    fn ids(&self) -> Vec<i32> {
        match self {
            Self::Single(id) => vec![*id],
            Self::Multiple(ids) => ids.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "telechat3".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_max_position_embeddings() -> usize {
    2048
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Kept as a raw JSON value rather than a typed struct: the YaRN reader
    /// takes the block as-is, and a typed struct would have to enumerate every
    /// vendor's optional fields (`mscale`, `mscale_all_dim`, ...) to avoid
    /// dropping one.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
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

    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// Build the YaRN frequency table, or `None` when the config asks for plain
    /// RoPE.
    ///
    /// The shared reader expects the rotary base inside the same block, but
    /// TeleChat3 keeps `rope_theta` at the top level of `config.json`, so it is
    /// injected here rather than defaulted by the reader (whose fallback is
    /// 500000, not this family's 1000000).
    pub(crate) fn yarn_rope(&self) -> Option<crate::models::mellum::YarnRope> {
        let scaling = self.rope_scaling.as_ref()?;
        let mut params = scaling.clone();
        if let Some(map) = params.as_object_mut() {
            map.entry("rope_theta")
                .or_insert_with(|| serde_json::json!(self.rope_theta));
        }
        compute_yarn_rope(self.head_dim(), &params)
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
    pub rope_base: f32,
    /// YaRN frequencies; `None` selects the plain rotation at `rope_base`.
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
    pub rope_mscale: f32,
}

impl Attention {
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

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        let (q, k) = match &self.rope_freqs {
            Some(freqs) => {
                // YaRN scales Q and K before the rotation; the factor is 1.0
                // whenever the config's mscale pair cancels out.
                let (q, k) = if (self.rope_mscale - 1.0).abs() > 1e-6 {
                    (
                        mlxcel_core::multiply_scalar(&q, self.rope_mscale),
                        mlxcel_core::multiply_scalar(&k, self.rope_mscale),
                    )
                } else {
                    (q, k)
                };
                (
                    mlxcel_core::fast_rope_with_freqs(&q, self.head_dim, false, 1.0, offset, freqs),
                    mlxcel_core::fast_rope_with_freqs(&k, self.head_dim, false, 1.0, offset, freqs),
                )
            }
            None => (
                mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset),
                mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset),
            ),
        };

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Generation calls `forward` with `mask == None`, so a multi-token
        // prefill has to build its own causal mask here or the prefill runs
        // bidirectionally (issues #991 / #999).
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
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        rope_freqs: Option<UniquePtr<MlxArray>>,
        rope_mscale: f32,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), group_size, bits)?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: args.rope_theta,
            rope_freqs,
            rope_mscale,
        })
    }
}

// Feed-forward.

pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
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

        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                group_size,
                bits,
            )?,
        })
    }
}

// Transformer block.

pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
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

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        rope_freqs: Option<UniquePtr<MlxArray>>,
        rope_mscale: f32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");

        let self_attn = Attention::from_weights(
            weights,
            args,
            &format!("{prefix}.self_attn"),
            rope_freqs,
            rope_mscale,
        )?;
        let mlp = MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?;

        let input_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.input_layernorm.weight"))?,
            args.rms_norm_eps,
        );
        let post_attention_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            args.rms_norm_eps,
        );

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// TeleChat3 model.

pub struct TeleChat3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub eos_token_ids: Vec<i32>,
}

impl TeleChat3Model {
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

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // One frequency table serves every layer, so it is built once and the
        // per-layer copies share its values.
        let yarn = args.yarn_rope();
        let rope_mscale = yarn.as_ref().map(|y| y.mscale).unwrap_or(1.0);

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let freqs = yarn.as_ref().map(|y| mlxcel_core::copy(&y.freqs));
            layers.push(DecoderLayer::from_weights(
                weights,
                args,
                i,
                freqs,
                rope_mscale,
            )?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            args.rms_norm_eps,
        );

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
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// Helper functions.

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.

impl LanguageModel for TeleChat3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        TeleChat3Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        TeleChat3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "telechat3_tests.rs"]
mod telechat3_tests;
