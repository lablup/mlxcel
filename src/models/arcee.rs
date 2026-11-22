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

//! Arcee AFM dense text models (`arcee`, `ArceeForCausalLM`).
//!
//! Reference: Hugging Face transformers
//! `src/transformers/models/arcee/modeling_arcee.py` (validated against
//! transformers 5.24.0). Neither mlx-lm nor mlx-vlm implements `arcee`, so the
//! maintainer accepted transformers as the parity reference (#1719);
//! `scripts/arcee_oracle_trace.py` scores the `logit_trace` chunk plan with it.
//!
//! Not to be confused with Arcee AFMoE / Trinity ([`super::afmoe`]) or Apple's
//! AFM-7 (`afm7`): the three share initials and nothing else.
//!
//! The decoder is Llama-shaped except for its MLP: GQA attention, RMSNorm
//! before attention and before the MLP, an untied head, Llama's weight key
//! names, and YaRN RoPE (`factor` 20 over an original 4096 context on
//! AFM-4.5B). The MLP has no gate: `down_proj(relu2(up_proj(x)))` with
//! `relu2(x) = relu(x)^2`, which is why AFM-4.5B ships only `mlp.up_proj` and
//! `mlp.down_proj` per layer and is 7.2x wide. Attention is the shared
//! [`llama3::Attention`], so YaRN, the fused QKV projection, cache-owned
//! attention dispatch (paged decode, KV quantization) and split-attention
//! batched decode are the Llama paths unchanged; only the block and the MLP
//! live here.
//!
//! # What is refused
//!
//! [`ArceeConfig::validate`] refuses, naming the field, what the shared
//! decoder would otherwise decode differently from transformers without any
//! error: a `hidden_act` other than `relu2`, a RoPE type the shared RoPE path
//! does not implement, a YaRN block without a usable `factor`, a YaRN
//! attention magnitude that differs from the one transformers derives, a YaRN
//! `truncate: false`, and geometry that would divide by zero or overflow.

use mlxcel_core::cache::{BatchedAttentionMetadata, SequenceId};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{pipeline_hint, relu_squared};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use super::llama3::{self, EosTokenId, RopeScaling};
use super::rope_utils::{RopeScalingKind, yarn_get_mscale};

/// Arcee `config.json`, restricted to the fields the dense path reads or must
/// screen. Defaults follow transformers' `ArceeConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct ArceeConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub quantization: Option<llama3::Quantization>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    /// RoPE keys transformers reads that [`RopeScaling`] does not carry,
    /// captured by the `Deserialize` impl below so every route that builds
    /// this config (directory loader and adapter path alike) screens them.
    #[serde(skip)]
    pub rope_screen: RopeScreen,
}

/// RoPE keys outside [`RopeScaling`] that [`ArceeConfig::validate`] screens.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RopeScreen {
    /// YaRN `truncate`; `None` means not declared (transformers: `true`).
    pub truncate: Option<bool>,
    /// `partial_rotary_factor`, top level or lifted from `rope_parameters`.
    pub partial_rotary_factor: Option<f64>,
    /// A top-level `original_max_position_embeddings`, which transformers
    /// prefers over the one inside the RoPE block.
    pub top_level_original_max_position_embeddings: Option<f64>,
    /// `type` and `rope_type` when both are declared and disagree:
    /// transformers reads `rope_type`, the shared path reads `type`.
    pub conflicting_types: Option<(String, String)>,
}

impl RopeScreen {
    fn from_lifted(value: &serde_json::Value) -> Self {
        use serde_json::Value;
        let block = ["rope_scaling", "rope_parameters"]
            .iter()
            .find_map(|key| value.get(*key).filter(|v| v.is_object()));
        let in_block = |key: &str| block.and_then(|b| b.get(key));
        let conflicting_types = match (
            in_block("type").and_then(Value::as_str),
            in_block("rope_type").and_then(Value::as_str),
        ) {
            (Some(a), Some(b)) if a != b => Some((a.to_string(), b.to_string())),
            _ => None,
        };
        Self {
            truncate: in_block("truncate").and_then(Value::as_bool),
            partial_rotary_factor: value
                .get("partial_rotary_factor")
                .or_else(|| in_block("partial_rotary_factor"))
                .and_then(Value::as_f64),
            top_level_original_max_position_embeddings: value
                .get("original_max_position_embeddings")
                .and_then(Value::as_f64),
            conflicting_types,
        }
    }
}

impl<'de> Deserialize<'de> for ArceeConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        mlxcel_core::rope_parameters::deserialize_lifting_rope_parameters(deserializer, |value| {
            let screen = RopeScreen::from_lifted(&value);
            let mut config = ArceeConfig::deserialize(value)?;
            config.rope_screen = screen;
            Ok(config)
        })
    }
}

fn default_model_type() -> String {
    "arcee".to_string()
}
/// transformers `ArceeConfig.hidden_act` default.
fn default_hidden_act() -> String {
    "relu2".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}

/// The RoPE types [`RopeScalingKind::resolve`] implements. Anything else
/// would fall back to the unscaled rotation with only a warning.
const IMPLEMENTED_ROPE_TYPES: &[&str] = &["default", "linear", "llama3", "yarn"];

/// Ceilings on untrusted scalars, in the style of `k2_horizon.rs`.
const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
const MAX_HEAD_DIM: usize = 1024;
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
const MAX_HIDDEN_SIZE: usize = 65_536;
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
const MAX_MAX_POSITION_EMBEDDINGS: usize = 1 << 22;
const MAX_VOCAB_SIZE: usize = 1 << 24;

impl ArceeConfig {
    /// Head width: the declared `head_dim`, else `hidden_size / heads`.
    ///
    /// Only valid after [`Self::validate`], which rejects zero heads.
    pub fn resolved_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Refuse a config the shared decoder would decode differently from
    /// transformers' `ArceeForCausalLM`.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_act != "relu2" {
            return Err(format!(
                "Arcee hidden_act ({}) is not supported: the Arcee MLP here implements only \
                 relu2 (squared ReLU)",
                self.hidden_act
            ));
        }
        self.validate_geometry()?;
        self.validate_rope()?;
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Arcee rms_norm_eps ({}) must be a finite positive number",
                self.rms_norm_eps
            ));
        }
        if let Some(q) = self.quantization.as_ref() {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("Arcee config.json: {e}"))?;
        }
        Ok(())
    }

    fn validate_rope(&self) -> Result<(), String> {
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Arcee rope_theta ({}) must be a finite positive number",
                self.rope_theta
            ));
        }
        let Some(spec) = self.rope_scaling.as_ref() else {
            return Ok(());
        };
        let rope_type = spec.rope_type();
        if !IMPLEMENTED_ROPE_TYPES.contains(&rope_type) {
            return Err(format!(
                "Arcee rope_scaling type {rope_type:?} is not supported: the shared RoPE path \
                 implements {IMPLEMENTED_ROPE_TYPES:?}"
            ));
        }
        if let Some((ty, rope_ty)) = self.rope_screen.conflicting_types.as_ref() {
            return Err(format!(
                "Arcee rope_scaling declares type {ty:?} and rope_type {rope_ty:?}: \
                 transformers reads rope_type and the shared RoPE path reads type"
            ));
        }
        if let Some(prf) = self.rope_screen.partial_rotary_factor
            && prf != 1.0
        {
            return Err(format!(
                "Arcee partial_rotary_factor ({prf}) is not supported: the shared attention \
                 rotates the full head"
            ));
        }
        if rope_type == "linear" || rope_type == "llama3" {
            // A block the shared path cannot use falls back to the unscaled
            // rotation with only a warning; refuse the obvious case here.
            return match spec.factor {
                Some(f) if f.is_finite() && f > 0.0 => Ok(()),
                other => Err(format!(
                    "Arcee {rope_type} rope_scaling needs a positive finite factor, got {other:?}"
                )),
            };
        }
        if rope_type != "yarn" {
            return Ok(());
        }
        // transformers derives a missing factor from the context ratio; the
        // shared path would fall back to the unscaled rotation instead.
        let factor = match spec.factor {
            Some(f) if f.is_finite() && f > 0.0 => f,
            other => {
                return Err(format!(
                    "Arcee YaRN rope_scaling needs a positive finite factor, got {other:?}"
                ));
            }
        };
        for (name, value) in [
            ("beta_fast", spec.beta_fast),
            ("beta_slow", spec.beta_slow),
            (
                "original_max_position_embeddings",
                spec.original_max_position_embeddings,
            ),
        ] {
            if let Some(v) = value
                && !(v.is_finite() && v > 0.0)
            {
                return Err(format!(
                    "Arcee YaRN rope_scaling.{name} ({v}) must be a positive finite number"
                ));
            }
        }
        if let (Some(fast), Some(slow)) = (spec.beta_fast, spec.beta_slow)
            && fast < slow
        {
            return Err(format!(
                "Arcee YaRN rope_scaling.beta_fast ({fast}) is below beta_slow ({slow}): the \
                 correction ramp would be inverted"
            ));
        }
        if let Some(top) = self.rope_screen.top_level_original_max_position_embeddings
            && spec
                .original_max_position_embeddings
                .is_some_and(|inner| (f64::from(inner) - top).abs() > 0.5)
        {
            return Err(format!(
                "Arcee original_max_position_embeddings ({top}) at the top level differs from \
                 the rope_scaling block's: transformers prefers the top-level value and the \
                 shared YaRN path reads the block's"
            ));
        }
        if self.rope_screen.truncate == Some(false) {
            return Err(
                "Arcee YaRN rope_scaling.truncate = false is not supported: the shared YaRN \
                 path always rounds the correction range outward"
                    .to_string(),
            );
        }
        let reference = transformers_yarn_attention_factor(
            factor,
            spec.attention_factor,
            spec.mscale,
            spec.mscale_all_dim,
        );
        let shared = yarn_get_mscale(factor, spec.mscale.unwrap_or(1.0))
            / yarn_get_mscale(factor, spec.mscale_all_dim.unwrap_or(0.0));
        let tolerance = 1e-6 * reference.abs().max(shared.abs());
        if !reference.is_finite() || (reference - shared).abs() > tolerance {
            return Err(format!(
                "Arcee YaRN attention factor: transformers derives {reference} from this \
                 rope_scaling block but the shared YaRN path would apply {shared}; this \
                 combination of attention_factor / mscale / mscale_all_dim is not supported"
            ));
        }
        Ok(())
    }

    /// Reject head counts and widths that would divide by zero, overflow the
    /// i32 casts in the decoder, or reach an MLX kernel with an odd rotary
    /// dimension.
    fn validate_geometry(&self) -> Result<(), String> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("intermediate_size", self.intermediate_size),
            ("num_attention_heads", self.num_attention_heads),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return Err(format!("Arcee {name} must be at least 1"));
            }
        }
        for (name, value, ceiling) in [
            (
                "num_hidden_layers",
                self.num_hidden_layers,
                MAX_NUM_HIDDEN_LAYERS,
            ),
            ("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE),
            (
                "intermediate_size",
                self.intermediate_size,
                MAX_INTERMEDIATE_SIZE,
            ),
            ("vocab_size", self.vocab_size, MAX_VOCAB_SIZE),
            (
                "num_attention_heads",
                self.num_attention_heads,
                MAX_NUM_ATTENTION_HEADS,
            ),
            (
                "num_key_value_heads",
                self.num_key_value_heads.unwrap_or(0),
                MAX_NUM_ATTENTION_HEADS,
            ),
            (
                "max_position_embeddings",
                self.max_position_embeddings.unwrap_or(0),
                MAX_MAX_POSITION_EMBEDDINGS,
            ),
        ] {
            if value > ceiling {
                return Err(format!("Arcee {name} ({value}) exceeds {ceiling}"));
            }
        }
        let kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        if kv_heads == 0 || !self.num_attention_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "Arcee num_attention_heads ({}) must be a positive multiple of \
                 num_key_value_heads ({kv_heads})",
                self.num_attention_heads
            ));
        }
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "Arcee hidden_size ({}) must be divisible by num_attention_heads ({}) when \
                 head_dim is not declared",
                self.hidden_size, self.num_attention_heads
            ));
        }
        let head_dim = self.resolved_head_dim();
        if head_dim < 2 || !head_dim.is_multiple_of(2) || head_dim > MAX_HEAD_DIM {
            return Err(format!(
                "Arcee head_dim ({head_dim}) must be even and between 2 and {MAX_HEAD_DIM}"
            ));
        }
        for (name, heads) in [
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", kv_heads),
        ] {
            let fits = heads
                .checked_mul(head_dim)
                .is_some_and(|width| i32::try_from(width).is_ok());
            if !fits {
                return Err(format!(
                    "Arcee {name} ({heads}) * head_dim ({head_dim}) does not fit in i32"
                ));
            }
        }
        Ok(())
    }

    /// Build the `llama3::ModelArgs` that drives the shared attention.
    ///
    /// A full struct literal on purpose (the pattern of
    /// `k2_horizon::K2HorizonConfig::to_llama3_args`): a field added to
    /// [`llama3::ModelArgs`] fails this file to compile instead of silently
    /// defaulting for Arcee.
    pub fn to_llama3_args(&self) -> llama3::ModelArgs {
        llama3::ModelArgs {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            head_dim: Some(self.resolved_head_dim()),
            num_key_value_heads: self.num_key_value_heads,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling.clone(),
            max_position_embeddings: self.max_position_embeddings,
            quantization: self.quantization.clone(),
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            eos_token_id: self.eos_token_id.clone(),
            // transformers rotates split halves (`rotate_half`).
            rope_traditional: false,
            checkpoint_label: None,
        }
    }
}

/// transformers' `_compute_yarn_parameters` attention factor: a declared
/// `attention_factor` wins; otherwise the `mscale` / `mscale_all_dim` ratio
/// when both are set and non-zero; otherwise `0.1 * ln(factor) + 1`.
pub(crate) fn transformers_yarn_attention_factor(
    factor: f32,
    attention_factor: Option<f32>,
    mscale: Option<f32>,
    mscale_all_dim: Option<f32>,
) -> f32 {
    if let Some(declared) = attention_factor {
        return declared;
    }
    match (mscale, mscale_all_dim) {
        (Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
            yarn_get_mscale(factor, m) / yarn_get_mscale(factor, all)
        }
        _ => yarn_get_mscale(factor, 1.0),
    }
}

/// Parse and screen an Arcee `config.json`.
pub fn parse_config(config_str: &str) -> Result<ArceeConfig, String> {
    let config: ArceeConfig = serde_json::from_str(config_str)
        .map_err(|e| format!("Failed to parse Arcee config.json: {e}"))?;
    config.validate()?;
    Ok(config)
}

/// The gate-free squared-ReLU MLP: `down_proj(relu(up_proj(x))^2)`.
pub struct ArceeMLP {
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl ArceeMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.down_proj
            .forward(&relu_squared(&self.up_proj.forward(x)))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &llama3::ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        if weights.get(&format!("{prefix}.gate_proj.weight")).is_some() {
            return Err(format!(
                "{prefix}.gate_proj is present, but the Arcee MLP is gate-free \
                 (down_proj(relu2(up_proj(x)))); this checkpoint is not an Arcee dense model"
            ));
        }
        Ok(Self {
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

/// One decoder layer: the shared Llama attention and the Arcee MLP.
pub struct ArceeBlock {
    pub self_attn: llama3::Attention,
    pub mlp: ArceeMLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl ArceeBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let (normed, h) =
            mlxcel_core::layers::fused_add_rms_norm(&self.post_attention_layernorm, &attn_out, x);
        mlxcel_core::add(&h, &self.mlp.forward(&normed))
    }

    /// Batched decode: batched norms, projections and MLP, per-sequence
    /// attention through the shared split-attention path.
    pub fn forward_batched(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let (q, k, v) = self.self_attn.qkv_proj.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(caches, seq_len, 0)
            .expect("valid arcee batched attention metadata");
        let attn_concat = mlxcel_core::batched_decode::forward_split_attention(
            &self.self_attn,
            &q,
            &k,
            &v,
            caches,
            &metadata,
            mask,
        );
        let attn_out = self.self_attn.o_proj.forward(&attn_concat);
        let (normed, h) =
            mlxcel_core::layers::fused_add_rms_norm(&self.post_attention_layernorm, &attn_out, x);
        mlxcel_core::add(&h, &self.mlp.forward(&normed))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &llama3::ModelArgs,
        layer_idx: usize,
        rope: &RopeScalingKind,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let self_attn = llama3::Attention::from_weights_with_rope(
            weights,
            args,
            &format!("{prefix}.self_attn"),
            rope,
        )?;
        let mlp = ArceeMLP::from_weights(weights, args, &format!("{prefix}.mlp"))?;
        let input_layernorm = RMSNorm::new(
            weight_copy(weights, &format!("{prefix}.input_layernorm.weight"))?,
            args.rms_norm_eps,
        );
        let post_attention_layernorm = RMSNorm::new(
            weight_copy(
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

fn weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

/// The Arcee dense decoder.
pub struct ArceeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<ArceeBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    eos_token_ids: Vec<i32>,
}

impl ArceeModel {
    /// The normalized hidden state `[B, L, hidden]` the LM head consumes.
    pub fn hidden_states(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_tokens.forward(input_ids),
        };
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }
        self.norm.forward(&h)
    }

    /// Logits `[B, 1, vocab]` for position `last_pos` only.
    pub fn last_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, input_embeddings, caches, mask);
        let row = mlxcel_core::generate::logits_at_position(&h, last_pos);
        self.lm_head.forward(&row)
    }

    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, None, caches, mask);
        self.lm_head.forward(&h)
    }

    /// Batched forward: batched embedding, norms, projections and MLP,
    /// per-sequence attention. Returns `[B, T, vocab]`.
    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for layer_idx in 0..self.layers.len() {
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();
            h = self.layers[layer_idx].forward_batched(&h, &mut layer_caches, mask);
        }
        self.lm_head.forward(&self.norm.forward(&h))
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load a checkpoint directory.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ArceeConfig), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = parse_config(&config_str)?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let mut args = config.to_llama3_args();
        args.set_checkpoint_label(model_dir);
        let model = Self::from_llama3_args(&weights, &args)?;
        Ok((model, config))
    }

    /// Build from already-loaded weights (adapter path).
    pub fn from_weights(weights: &WeightMap, config: &ArceeConfig) -> Result<Self, String> {
        config.validate()?;
        Self::from_llama3_args(weights, &config.to_llama3_args())
    }

    fn from_llama3_args(weights: &WeightMap, args: &llama3::ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        // One frequency table for the whole stack.
        let rope = args.rope_scaling_kind();
        let layers = (0..args.num_hidden_layers)
            .map(|i| ArceeBlock::from_weights(weights, args, i, &rope))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = RMSNorm::new(
            weight_copy(weights, "model.norm.weight")?,
            args.rms_norm_eps,
        );
        let lm_head = if args.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
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

impl LanguageModel for ArceeModel {
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ArceeModel::forward(self, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, input_embeddings, caches, mask);
        self.lm_head.forward(&h)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        Some(self.last_logits(input_ids, input_embeddings, caches, mask, last_pos))
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        ArceeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if batch_caches.is_empty() {
            return mlxcel_core::zeros(&[0, 1, 1], mlxcel_core::dtype::FLOAT32);
        }
        self.forward_batched_impl(input_ids, batch_caches, mask)
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }

    fn supports_paged_decode_backend(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "arcee_tests.rs"]
mod tests;
