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

//! Aleph Alpha Kolibri-1 (`kolibri1`, `Kolibri1ForCausalLM`).
//!
//! Ported from mlx-vlm's `mlx_vlm/models/kolibri1/language.py`.
//!
//! Structurally this is [`crate::models::afmoe`] without the attention output
//! gate, so the attention block, the shared-expert MLP, the prefill masks and
//! the weight validators are AFMoE's, reused rather than restated:
//!
//! - `layer_types` interleaves `sliding_attention` and `full_attention` (four
//!   sliding then one full, ten times over on the released 50-layer config).
//!   Sliding layers hold a `RotatingKVCache(max_size = sliding_window, keep =
//!   0)` and full layers a `KVCache`, carried in model-owned state through the
//!   gemma3 `Cache` enum as AFMoE does.
//! - **Full-attention layers are NoPE.** Upstream builds `self.rope` only on
//!   sliding layers, so rotating a full layer loads every tensor and is
//!   invisible at position 0.
//! - Per-head Q/K RMSNorm before RoPE, and four norms per layer in a sandwich:
//!   `h = x + post_attn_norm(attn(input_layernorm(x)))`,
//!   `out = h + post_ffn_norm(moe(post_attention_layernorm(h)))`.
//! - Every layer is MoE: 384 routed experts top-6 plus one SwiGLU shared
//!   expert added unconditionally. The router selects on **raw logits plus a
//!   correction bias** (not sigmoid plus bias) and weights with the sigmoid of
//!   the unbiased logits; see [`crate::models::kolibri1_moe`].
//! - Logits are cast to f32 when `head_dtype == "float32"` (the default).
//!
//! The released checkpoint ships its 384 experts unstacked and its selection
//! bias under `moe.router.expert_bias`; [`crate::models::kolibri1_weights`]
//! renames and stacks them at load, then refuses a stack with fewer planes
//! than `num_experts`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::models::afmoe::{
    Attention, AttentionGeometry, Quantization, TokenIdField, hybrid_prefill_masks,
    rms_norm_from_weights,
};
use crate::models::gemma3::{Cache, CacheInterface};
use crate::models::gpt2::validate_embedding_table;
use crate::models::kolibri1_moe::Kolibri1Moe;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};

const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
const MAX_HIDDEN_SIZE: usize = 65_536;
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
const MAX_HEAD_DIM: usize = 8192;
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
const MAX_VOCAB_SIZE: usize = 1 << 24;
const MAX_NUM_EXPERTS: usize = 4096;

fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_head_dtype() -> String {
    "float32".to_string()
}

/// Kolibri-1 `config.json`. The architecture fields upstream's dataclass
/// requires are required here too.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub sliding_window: usize,
    /// Per-layer attention kind, one `"sliding_attention"` or
    /// `"full_attention"` entry per layer. Upstream refuses any other length
    /// or value in `__post_init__`, and so does [`ModelArgs::validate`].
    pub layer_types: Vec<String>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// `rope_parameters.rope_theta` overrides this when present (lifted by
    /// `impl_rope_lifting_deserialize!`, as upstream's `__post_init__` does).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_head_dtype")]
    pub head_dtype: String,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

mlxcel_core::impl_rope_lifting_deserialize!(ModelArgs);

impl ModelArgs {
    /// Whether layer `layer_idx` uses sliding-window attention. Read from the
    /// list, which [`Self::validate`] guarantees covers every layer.
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|t| t == "sliding_attention")
    }

    /// First full-attention layer, whose cache sizes the global prefill mask.
    pub fn full_attention_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| !self.is_sliding_layer(i))
    }

    /// First sliding layer, whose cache sizes the windowed prefill mask.
    pub fn sliding_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| self.is_sliding_layer(i))
    }

    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }

    /// Whether the logits are cast to f32 after the head.
    pub fn head_is_f32(&self) -> bool {
        self.head_dtype == "float32"
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(TokenIdField::Single(id)) => vec![*id],
            Some(TokenIdField::Multiple(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }

    pub(crate) fn attention_geometry(&self) -> AttentionGeometry {
        AttentionGeometry {
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            group_size: self.group_size(),
            bits: self.bits(),
        }
    }

    /// Reject a `config.json` that cannot describe a real Kolibri-1 before any
    /// field sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        let bounded = |name: &str, value: usize, max: usize| {
            if value == 0 || value > max {
                Err(format!(
                    "Kolibri-1 {name} ({value}) must be between 1 and {max}"
                ))
            } else {
                Ok(())
            }
        };
        bounded(
            "num_hidden_layers",
            self.num_hidden_layers,
            MAX_NUM_HIDDEN_LAYERS,
        )?;
        bounded("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE)?;
        bounded(
            "num_attention_heads",
            self.num_attention_heads,
            MAX_NUM_ATTENTION_HEADS,
        )?;
        bounded("head_dim", self.head_dim, MAX_HEAD_DIM)?;
        bounded("vocab_size", self.vocab_size, MAX_VOCAB_SIZE)?;
        bounded("num_experts", self.num_experts, MAX_NUM_EXPERTS)?;
        bounded(
            "moe_intermediate_size",
            self.moe_intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        bounded(
            "shared_expert_intermediate_size",
            self.shared_expert_intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        bounded(
            "max_position_embeddings",
            self.max_position_embeddings,
            1 << 24,
        )?;
        if self.num_key_value_heads == 0
            || self.num_key_value_heads > self.num_attention_heads
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "Kolibri-1 num_key_value_heads ({}) must divide num_attention_heads ({})",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "Kolibri-1 layer_types has {} entries for {} layers; upstream requires one \
                 entry per hidden layer, and a mismatched list would silently change which \
                 layers slide",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for (idx, kind) in self.layer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "sliding_attention" | "full_attention") {
                return Err(format!(
                    "Kolibri-1 layer_types[{idx}] is {kind:?}; only \"sliding_attention\" and \
                     \"full_attention\" are supported"
                ));
            }
        }
        crate::models::model_owned::cache_window(
            "Kolibri-1",
            self.sliding_window,
            self.sliding_index().is_some(),
        )?;
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(format!(
                "Kolibri-1 num_experts_per_tok ({}) must be between 1 and num_experts ({})",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        if self.hidden_act != "silu" {
            return Err(format!(
                "Kolibri-1 hidden_act {:?} is not supported; the experts are SwiGLU",
                self.hidden_act
            ));
        }
        if self.attention_bias {
            return Err(
                "Kolibri-1 attention_bias = true is not supported: no released checkpoint ships \
                 attention biases and this loader does not validate them"
                    .to_string(),
            );
        }
        if !matches!(self.head_dtype.as_str(), "float32" | "bfloat16" | "float16") {
            return Err(format!(
                "Kolibri-1 head_dtype {:?} is not one of float32, bfloat16, float16",
                self.head_dtype
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Kolibri-1 rope_theta ({}) must be a finite positive number",
                self.rope_theta
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "Kolibri-1 head_dim ({}) must be even; the sliding layers rotate channel pairs",
                self.head_dim
            ));
        }
        if let Some(scaling) = self.rope_scaling.as_ref().filter(|s| !s.is_empty()) {
            let kind = scaling
                .get("rope_type")
                .or_else(|| scaling.get("type"))
                .and_then(serde_json::Value::as_str);
            if kind != Some("default") {
                return Err(format!(
                    "Kolibri-1 rope_scaling ({scaling:?}) is not implemented; only an absent, \
                     empty or \"default\" block is accepted"
                ));
            }
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Kolibri-1 rms_norm_eps ({}) must be a finite positive number",
                self.rms_norm_eps
            ));
        }
        if let Some(q) = self.quantization.as_ref() {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("Kolibri-1 config.json: {e}"))?;
        }
        Ok(())
    }
}

/// One Kolibri-1 block: a gateless AFMoE attention and a routed-plus-shared
/// MoE, each branch normalized on its output before it joins the residual.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: Kolibri1Moe,
    pub input_layernorm: RMSNorm,
    pub post_attn_norm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub post_ffn_norm: RMSNorm,
    pub uses_sliding: bool,
}

impl DecoderLayer {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x), cache, mask);
        let h = mlxcel_core::add(x, &self.post_attn_norm.forward(&attn));
        let ffn = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &self.post_ffn_norm.forward(&ffn))
    }

    pub(crate) fn forward_batched_decode(
        &self,
        x: &MlxArray,
        rows: &mut [&mut Cache],
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward_batched_decode(&self.input_layernorm.forward(x), rows);
        let h = mlxcel_core::add(x, &self.post_attn_norm.forward(&attn));
        let ffn = self
            .mlp
            .forward_batched_decode(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &self.post_ffn_norm.forward(&ffn))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let uses_sliding = args.is_sliding_layer(layer_idx);
        let norm = |leaf: &str| {
            rms_norm_from_weights(
                weights,
                &format!("{prefix}.{leaf}.weight"),
                args.rms_norm_eps,
            )
        };
        // Only sliding layers rotate; there is no attention output gate.
        let mut self_attn = Attention::load(
            weights,
            &format!("{prefix}.self_attn"),
            &args.attention_geometry(),
            uses_sliding,
            false,
        )?;
        if uses_sliding {
            self_attn.prefill_window = i32::try_from(args.sliding_window).unwrap_or(i32::MAX);
        }
        Ok(Self {
            self_attn,
            mlp: Kolibri1Moe::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            input_layernorm: norm("input_layernorm")?,
            post_attn_norm: norm("post_attn_norm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            post_ffn_norm: norm("post_ffn_norm")?,
            uses_sliding,
        })
    }
}

/// Aleph Alpha Kolibri-1.
pub struct Kolibri1Model {
    pub embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    head_f32: bool,
    pub(crate) sliding_window: i32,
    full_attention_index: Option<usize>,
    sliding_index: Option<usize>,
    pub(crate) eos_token_ids: Vec<i32>,
    /// Sliding layers hold a `RotatingKVCache` and full layers a `KVCache`,
    /// which does not fit the trait's homogeneous `&mut [KVCache]`, so the
    /// model owns its heterogeneous cache and persists it across calls.
    pub(crate) sequence_state: ModelOwnedSequenceState<Cache>,
    pub(crate) kv_cache_layer_modes: KvCacheLayerModes,
}

impl Kolibri1Model {
    fn project(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(h);
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        };
        if self.head_f32 {
            mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32)
        } else {
            logits
        }
    }

    pub(crate) fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            let mut rows: Vec<_> = caches.iter_mut().map(|state| &mut state[i]).collect();
            h = layer.forward_batched_decode(&h, &mut rows);
        }
        self.project(&h)
    }

    /// Single-sequence forward projecting every position (`row == None`) or
    /// only position `row` (`[B, 1, vocab]`, the prefill's sampled row).
    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        let (full_mask, sliding_mask) = hybrid_prefill_masks(
            &h,
            caches,
            self.full_attention_index,
            self.sliding_index,
            self.sliding_window,
            |idx| {
                self.layers
                    .get(idx)
                    .map(|layer| layer.self_attn.prefill_attention())
            },
        );
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.uses_sliding {
                sliding_mask.as_deref()
            } else {
                full_mask.as_deref()
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
        }
        let h = mlxcel_core::generate::select_row(h, row);
        self.project(&h)
    }

    pub(crate) fn make_internal_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                if layer.uses_sliding {
                    let mode = self
                        .kv_cache_layer_modes
                        .rotating_mode_for_layer(layer_idx, self.sliding_window);
                    Cache::Rotating(RotatingKVCache::new_with_mode(self.sliding_window, mode))
                } else {
                    let mode = self.kv_cache_layer_modes.mode_for_layer(layer_idx);
                    Cache::Standard(KVCache::new_with_mode(mode))
                }
            })
            .collect()
    }

    /// The single-sequence forward against the model-owned caches of
    /// `seq_id`. An id-less multi-token forward starts an independent prompt.
    pub(crate) fn forward_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_internal_caches());
        }
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| self.forward_rows(input, caches, row),
        )
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        // Reject an impossible config before reading 45 GB of weights.
        args.validate()?;
        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&mut weights, &args)?;
        Ok((model, args))
    }

    /// Build the model, first renaming the selection bias and stacking the
    /// per-expert planes in place ([`crate::models::kolibri1_weights`]). The
    /// pass is idempotent, so an already-stacked map goes through unchanged.
    pub fn from_weights(weights: &mut WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        crate::models::kolibri1_weights::sanitize_weights(weights, args);
        crate::models::kolibri1_weights::validate_weights(weights, args)?;

        let group_size = args.group_size();
        let bits = args.bits();
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        validate_embedding_table(
            &embed_tokens,
            "model.embed_tokens",
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;
        let sliding_window = crate::models::model_owned::cache_window(
            "Kolibri-1",
            args.sliding_window,
            args.sliding_index().is_some(),
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::from_weights(weights, args, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = rms_norm_from_weights(weights, "model.norm.weight", args.rms_norm_eps)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };
        let internal_caches = layers
            .iter()
            .map(|layer| {
                if layer.uses_sliding {
                    Cache::Rotating(RotatingKVCache::new(sliding_window))
                } else {
                    Cache::Standard(KVCache::new())
                }
            })
            .collect();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            head_f32: args.head_is_f32(),
            sliding_window,
            full_attention_index: args.full_attention_index(),
            sliding_index: args.sliding_index(),
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
        })
    }
}

#[cfg(test)]
#[path = "kolibri1_tests.rs"]
mod tests;
