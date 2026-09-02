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

//! Youtu language model (the text backbone of Youtu-VL, and the whole model for
//! text-only Youtu-LLM).
//!
//! Faithful port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/language.py.
//! The vendor's text-only decoder is the same architecture; see
//! https://huggingface.co/tencent/Youtu-LLM-2B/blob/main/modeling_youtu.py (YoutuMLAttention).
//!
//! Architecture summary:
//! - Multi-Latent Attention (MLA) — identical to DeepSeek-V3's MLA layout
//!   (Q LoRA via `q_a_proj` + `q_a_layernorm` + `q_b_proj`; KV via
//!   `kv_a_proj_with_mqa` + `kv_a_layernorm` + `embed_q` / `unembed_out`).
//! - SwiGLU dense MLP only (the open `Youtu-VL-8B` config sets
//!   `n_routed_experts=None`, so MoE layers never trigger; we therefore
//!   keep the language model strictly dense for the standard checkpoint
//!   to avoid pulling in MoE code paths the released weights don't use).
//! - RMSNorm pre/post attention.
//! - Tied word embeddings (`tie_word_embeddings = True`).
//! - Traditional/interleaved RoPE (`rope_traditional = True`).
//!
//! Reuse policy:
//! - Attention module is reused from [`super::deepseek_v3::DeepSeekV3Attention`]
//!   verbatim. The MLA layout (weight names, head decomposition, decode vs.
//!   prefill paths) is byte-for-byte identical between Youtu-VL and DeepSeek-V3,
//!   so re-implementing it would duplicate ~150 lines of subtle code. We bridge
//!   by building a [`super::deepseek_v3::DeepSeekV3Config`] purely as a carrier
//!   struct for the attention constructor.
//! - `UnifiedLinear`, `UnifiedEmbedding`, `RMSNorm`, and `KVCache` come from
//!   `mlxcel_core::layers`.
//!
//! Used by: Youtu-VL VLM (`vision::youtu_vl::YoutuVLModel`), text-only
//! Youtu-LLM (`ModelType::YoutuLLM` through `loading::nonstandard`).

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

use super::deepseek_v3::DeepSeekV3Attention;

#[path = "youtu_vl_lm_config.rs"]
mod youtu_vl_lm_config;
#[path = "youtu_vl_lm_sanitize.rs"]
mod youtu_vl_lm_sanitize;

pub use youtu_vl_lm_config::{QuantizationConfig, YoutuTextConfig};
pub use youtu_vl_lm_sanitize::sanitize_text_weights;

// Dense SwiGLU MLP — identical layout to other Llama-family MLPs but kept local
// so we can pin it to Youtu-VL's `mlp_bias` flag and keep the language module
// self-contained. Only the dense path is supported; the released `youtu_vl`
// checkpoint never enters the MoE branch.
pub struct YoutuMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl YoutuMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &YoutuTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

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

// Decoder layer: pre-norm MLA + pre-norm SwiGLU MLP with residuals.
pub struct YoutuDecoderLayer {
    pub self_attn: DeepSeekV3Attention,
    pub mlp: YoutuMLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl YoutuDecoderLayer {
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
        config: &YoutuTextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let ds_cfg = config.to_deepseek_v3_config();
        // The shared MLA attention defaults to the interleaved pair layout that
        // every DeepSeek backbone uses; Youtu carries the choice in its config,
        // so the flag has to reach the rope call rather than being assumed.
        let self_attn =
            DeepSeekV3Attention::from_weights(weights, &ds_cfg, &format!("{}.self_attn", prefix))?
                .with_rope_traditional(config.rope_is_interleaved());

        let mlp = YoutuMLP::from_weights(weights, config, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, config.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attn_norm_weight, config.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Top-level language model.
pub struct YoutuLanguageModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<YoutuDecoderLayer>,
    pub norm: RMSNorm,
    /// `None` when `tie_word_embeddings = true` (standard Youtu-VL).
    pub lm_head: Option<UnifiedLinear>,
    pub config: YoutuTextConfig,
    /// EOS ids parsed from the loader (empty vec if not provided).
    pub eos_token_ids: Vec<i32>,
}

impl YoutuLanguageModel {
    /// Build the language model from a fully-sanitized weight map.
    ///
    /// The caller is responsible for running [`sanitize_text_weights`] before
    /// invoking this.
    pub fn from_weights(weights: &WeightMap, config: &YoutuTextConfig) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = YoutuDecoderLayer::from_weights(weights, config, i)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        // tie_word_embeddings: Python `sanitize` drops `language_model.lm_head.weight`
        // entirely when tied; here the weight map has already been stripped of the
        // `language_model.` prefix, so the head key (if present) is `lm_head.*`.
        let lm_head = if config.tie_word_embeddings {
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
            config: config.clone(),
            eos_token_ids: Vec::new(),
        })
    }

    pub fn with_eos_token_ids(mut self, ids: Vec<i32>) -> Self {
        self.eos_token_ids = ids;
        self
    }

    /// Forward pass shared by the `LanguageModel` trait and the VLM wrapper.
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        // Preserve any caller-provided mask for padded prefill, but let the
        // attention stack use its implicit causal path otherwise (matches the
        // DeepSeek-V3 dispatch pattern).
        let mask_array = mask.map(mlxcel_core::copy);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask_array.as_deref());
        }

        let h = self.norm.forward(&h);

        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    /// Helper that gets text embeddings directly (used by the VLM wrapper to
    /// construct merged embeddings before calling `forward_with_embeddings`).
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn make_caches_impl(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }
}

impl LanguageModel for YoutuLanguageModel {
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
        self.make_caches_impl()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        if self.eos_token_ids.is_empty() {
            // Conservative fallback when the loader did not parse explicit
            // eos ids — matches the behaviour of other text loaders that
            // also leave this empty.
            Vec::new()
        } else {
            self.eos_token_ids.clone()
        }
    }
}

// Standalone text-only loader. This is the directory route for the text-only
// Youtu-LLM checkpoint (`ModelType::YoutuLLM`, issue #1371); it also serves
// ad-hoc CLI runs that point at a Youtu-VL directory and want only its text
// tower.
impl YoutuLanguageModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, YoutuTextConfig), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: YoutuTextConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        config.validate_rope_scaling()?;

        let raw_config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let eos_token_ids = parse_eos_token_ids(&raw_config);

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = sanitize_text_weights(weights, &config)?;

        let model = Self::from_weights(&weights, &config)?.with_eos_token_ids(eos_token_ids);
        Ok((model, config))
    }
}

/// Read `eos_token_id` from a checkpoint `config.json`, accepting both the
/// single-number and the array spelling.
///
/// The directory loaders that build a `LoadedModel` merge this with
/// `generation_config.json` / `tokenizer_config.json` through
/// `crate::loading::read_eos_token_ids`; parsing it here means the model itself
/// reports a stop id even when it is constructed outside that path.
fn parse_eos_token_ids(config: &serde_json::Value) -> Vec<i32> {
    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n
            .as_i64()
            .and_then(|id| i32::try_from(id).ok())
            .map(|id| vec![id])
            .unwrap_or_default(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .filter_map(|n| i32::try_from(n).ok())
            .collect(),
        _ => Vec::new(),
    }
}

// Helper.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

#[cfg(test)]
#[path = "youtu_vl_lm_tests.rs"]
mod tests;
