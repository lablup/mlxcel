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

//! [`LagunaDFlashConfig`]: the `config.json` contract of the Poolside Laguna
//! DFlash drafters (`poolside/Laguna-XS-2.1-DFlash`, `Laguna-S-2.1-DFlash`).
//!
//! The published checkpoints declare `model_type: "laguna"`,
//! `architectures: ["DFlashLagunaForCausalLM"]` and a nested `dflash_config`
//! block. Every field the loader reads is listed on the struct; the
//! validation in [`LagunaDFlashConfig::validate`] rejects the shapes the
//! drafter forward cannot represent (a non-causal block, a full-attention
//! layer, a per-element gate, a draft vocabulary that differs from the
//! target's) with a typed error before any weight is read.

use serde::Deserialize;

/// `architectures[0]` of every published Laguna DFlash drafter.
pub const LAGUNA_DFLASH_ARCHITECTURE: &str = "DFlashLagunaForCausalLM";

/// `model_type` the Laguna drafters declare (the same string as the target).
pub const LAGUNA_MODEL_TYPE: &str = "laguna";

const SLIDING_ATTENTION: &str = "sliding_attention";

/// Nested `dflash_config` block.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct DFlashBlock {
    #[serde(default = "default_block_size")]
    block_size: usize,
    mask_token_id: i32,
    num_target_layers: usize,
    target_layer_ids: Vec<usize>,
    #[serde(default = "default_true")]
    causal: bool,
}

/// Raw top-level fields, before validation.
#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    vocab_size: usize,
    #[serde(default)]
    draft_vocab_size: Option<usize>,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default)]
    sliding_windows: Option<Vec<usize>>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    gating: serde_json::Value,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    num_experts: usize,
    dflash_config: DFlashBlock,
}

fn default_block_size() -> usize {
    16
}
fn default_true() -> bool {
    true
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f64 {
    500_000.0
}
fn default_sliding_window() -> usize {
    512
}

/// Validated Laguna DFlash drafter configuration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LagunaDFlashConfig {
    pub hidden_size: usize,
    /// SwiGLU inner width; checked against `mlp.gate_proj` at load.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    /// Target vocabulary the drafter borrows `embed_tokens` and `lm_head` from.
    pub vocab_size: usize,
    /// Plain RoPE base applied to all `head_dim` dims (no `rope_parameters`).
    pub rope_theta: f32,
    /// Per-layer sliding window; `[sliding_window; num_hidden_layers]` when the
    /// checkpoint omits `sliding_windows` (XS 2.1 does).
    pub sliding_windows: Vec<usize>,
    /// Draft block length: one bonus plus `block_size - 1` masked positions.
    pub block_size: usize,
    /// Token id stamped into the masked positions (`〈|MASK|〉`, id 12).
    pub mask_token_id: i32,
    /// Layer count of the target; must equal the target's `num_layers()`.
    pub num_target_layers: usize,
    /// 0-based target layers whose post-block residual stream feeds the
    /// drafter (`[1, 13, 25, 33, 39]` on XS 2.1).
    pub target_layer_ids: Vec<usize>,
}

impl LagunaDFlashConfig {
    /// Parse and validate a Laguna DFlash `config.json`.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let raw: RawConfig = serde_json::from_value(value.clone())
            .map_err(|e| format!("Failed to parse Laguna DFlash config: {e}"))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawConfig) -> Result<Self, String> {
        if let Some(model_type) = raw.model_type.as_deref()
            && model_type != LAGUNA_MODEL_TYPE
        {
            return Err(format!(
                "Laguna DFlash drafter: model_type must be {LAGUNA_MODEL_TYPE:?}, got {model_type:?}"
            ));
        }
        if !raw.architectures.is_empty()
            && !raw
                .architectures
                .iter()
                .any(|a| a == LAGUNA_DFLASH_ARCHITECTURE)
        {
            return Err(format!(
                "Laguna DFlash drafter: architectures {:?} does not include \
                 {LAGUNA_DFLASH_ARCHITECTURE:?}",
                raw.architectures
            ));
        }
        let layers = raw.num_hidden_layers;
        if layers == 0 {
            return Err("Laguna DFlash drafter: num_hidden_layers must be at least 1".into());
        }
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads.max(1));
        if head_dim < 2 || raw.num_attention_heads == 0 || raw.num_key_value_heads == 0 {
            return Err(format!(
                "Laguna DFlash drafter: invalid attention geometry (heads {}, kv heads {}, \
                 head_dim {head_dim})",
                raw.num_attention_heads, raw.num_key_value_heads
            ));
        }
        if !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        {
            return Err(format!(
                "Laguna DFlash drafter: {} query heads do not form whole GQA groups over {} \
                 key/value heads",
                raw.num_attention_heads, raw.num_key_value_heads
            ));
        }
        if raw.rope_parameters.as_ref().is_some_and(|v| v.is_object()) {
            return Err(
                "Laguna DFlash drafter: rope_parameters is not supported; the published \
                 drafters use plain RoPE over all head dims"
                    .into(),
            );
        }
        if raw.attention_bias {
            return Err(
                "Laguna DFlash drafter: attention_bias must be false; the published drafters \
                 carry no projection biases and the forward has no bias path"
                    .into(),
            );
        }
        if raw.num_experts != 0 {
            return Err(format!(
                "Laguna DFlash drafter: num_experts must be 0 (dense MLP), got {}",
                raw.num_experts
            ));
        }
        let gating_ok = match &raw.gating {
            serde_json::Value::String(s) => s == "per-head",
            serde_json::Value::Bool(true) => true,
            _ => false,
        };
        if !gating_ok {
            return Err(format!(
                "Laguna DFlash drafter: gating must be \"per-head\", got {}",
                raw.gating
            ));
        }
        match &raw.layer_types {
            Some(types) => {
                if types.len() != layers {
                    return Err(format!(
                        "Laguna DFlash drafter: layer_types has {} entries for {layers} layers",
                        types.len()
                    ));
                }
                if let Some((i, other)) = types
                    .iter()
                    .enumerate()
                    .find(|(_, t)| t.as_str() != SLIDING_ATTENTION)
                {
                    return Err(format!(
                        "Laguna DFlash drafter: layer_types[{i}] = {other:?}; every drafter \
                         layer must be {SLIDING_ATTENTION:?}"
                    ));
                }
            }
            None => {
                return Err("Laguna DFlash drafter: layer_types is required".into());
            }
        }
        let sliding_windows = match raw.sliding_windows {
            Some(windows) => {
                if windows.len() != layers {
                    return Err(format!(
                        "Laguna DFlash drafter: sliding_windows has {} entries for {layers} \
                         layers",
                        windows.len()
                    ));
                }
                windows
            }
            None => vec![raw.sliding_window; layers],
        };
        if let Some(w) = sliding_windows.iter().find(|w| **w < 2) {
            return Err(format!(
                "Laguna DFlash drafter: sliding window {w} is too small; the block needs at \
                 least one context position inside the window"
            ));
        }
        let draft_vocab = raw.draft_vocab_size.unwrap_or(raw.vocab_size);
        if draft_vocab != raw.vocab_size {
            return Err(format!(
                "Laguna DFlash drafter: draft_vocab_size ({draft_vocab}) must equal vocab_size \
                 ({}); the drafter shares the target lm_head",
                raw.vocab_size
            ));
        }
        let block = raw.dflash_config;
        if !block.causal {
            return Err(
                "Laguna DFlash drafter: dflash_config.causal must be true (the drafter block \
                 is causal inside the proposal)"
                    .into(),
            );
        }
        if block.block_size < 2 {
            return Err(format!(
                "Laguna DFlash drafter: dflash_config.block_size ({}) must be at least 2",
                block.block_size
            ));
        }
        if block.mask_token_id < 0 || block.mask_token_id as usize >= raw.vocab_size {
            return Err(format!(
                "Laguna DFlash drafter: mask_token_id {} is outside the vocabulary (0..{})",
                block.mask_token_id, raw.vocab_size
            ));
        }
        if block.target_layer_ids.len() != layers {
            return Err(format!(
                "Laguna DFlash drafter: target_layer_ids has {} entries but the drafter has \
                 {layers} layers (one captured target layer per drafter layer)",
                block.target_layer_ids.len()
            ));
        }
        if !block.target_layer_ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(format!(
                "Laguna DFlash drafter: target_layer_ids {:?} must be strictly increasing",
                block.target_layer_ids
            ));
        }
        if let Some(id) = block
            .target_layer_ids
            .iter()
            .find(|id| **id >= block.num_target_layers)
        {
            return Err(format!(
                "Laguna DFlash drafter: target layer id {id} is outside num_target_layers {}",
                block.num_target_layers
            ));
        }
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            vocab_size: raw.vocab_size,
            rope_theta: raw.rope_theta as f32,
            sliding_windows,
            block_size: block.block_size,
            mask_token_id: block.mask_token_id,
            num_target_layers: block.num_target_layers,
            target_layer_ids: block.target_layer_ids,
        })
    }

    /// Check the drafter against the target it will be bound to.
    ///
    /// `target_layers` is the target's `num_layers()`, `target_vocab` the row
    /// count of its embedding table.
    pub fn validate_target(&self, target_layers: usize, target_vocab: usize) -> Result<(), String> {
        if target_layers != self.num_target_layers {
            return Err(format!(
                "Laguna DFlash drafter was trained against a {}-layer target but the loaded \
                 target has {target_layers} layers",
                self.num_target_layers
            ));
        }
        if target_vocab != self.vocab_size {
            return Err(format!(
                "Laguna DFlash drafter vocab_size {} does not match the target vocabulary \
                 {target_vocab}",
                self.vocab_size
            ));
        }
        Ok(())
    }

    /// Whether `path/config.json` names the Laguna DFlash drafter; `false`
    /// when the file is missing or unparseable (the loader's error to report).
    pub fn is_laguna_dflash_dir(path: &std::path::Path) -> bool {
        let Ok(bytes) = std::fs::read(path.join("config.json")) else {
            return false;
        };
        let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return false;
        };
        Self::is_laguna_dflash_config(&config)
    }

    /// Whether a parsed `config.json` names the Laguna DFlash drafter
    /// (`architectures: ["DFlashLagunaForCausalLM"]`, or `model_type: laguna`
    /// with a `dflash_config` block).
    pub fn is_laguna_dflash_config(config: &serde_json::Value) -> bool {
        let model_type = config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|m| m == LAGUNA_MODEL_TYPE);
        let has_block = config
            .get("dflash_config")
            .is_some_and(serde_json::Value::is_object);
        let declares_arch = config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|archs| {
                archs
                    .iter()
                    .any(|a| a.as_str() == Some(LAGUNA_DFLASH_ARCHITECTURE))
            });
        declares_arch || (model_type && has_block)
    }
}
