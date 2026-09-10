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

//! `config.json` of the Muse Glimmer assistant drafter (issue #1343).
//!
//! The published `meta-models/Muse-Glimmer-30B-assistant` config is flat:
//! `block_size`, `mask_token_id` and `target_layer_ids` sit at the top level
//! and there is no `dflash_config` block, no `num_target_layers` and no
//! `vocab_size`. [`MuseAssistantConfig::from_json`] also accepts the nested
//! `dflash_config` form the Qwen 3.5 DFlash and LFM2 DSpark drafters use, so
//! a re-export of the checkpoint through either tool loads too. The two
//! missing keys are `Option`s that are checked against the bound target at
//! bind time, where the pairing is verified anyway; defaulting them to zero
//! and validating `target_layer_ids[i] < num_target_layers` would reject the
//! real checkpoint.

use std::path::Path;

use serde::Deserialize;

/// `model_type` the published assistant checkpoint declares.
pub const MUSE_ASSISTANT_MODEL_TYPE: &str = "muse_glimmer_assistant";

/// `architectures[0]` the published assistant checkpoint declares.
pub const MUSE_ASSISTANT_ARCHITECTURE: &str = "MuseGlimmerAssistantModel";

/// The one layer type the drafter supports; every published layer is one.
pub const SLIDING_ATTENTION: &str = "sliding_attention";

/// Verify width the DFlash round loop starts a Muse Glimmer run at, before
/// its throughput comparator has measured the published 16-row block
/// against it (bonus row included, so three proposals). Measured on an M5
/// Max with `Muse-Glimmer-30B-4bit`: at 16 rows natural-text requests
/// accept 1.9 proposals per round and decode at 13 to 18 tok/s against 18
/// tok/s classic, while 4 rows accept 1.4 per round and decode at 34 to 41
/// tok/s; repetitive text accepts 14 of 15 at 16 rows (94 tok/s), which is
/// what the comparator widens for.
pub const MUSE_ASSISTANT_INITIAL_BLOCK_SIZE: usize = 4;

fn default_model_type() -> String {
    MUSE_ASSISTANT_MODEL_TYPE.to_string()
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_sliding_window() -> usize {
    2048
}

fn default_block_size() -> usize {
    16
}

fn default_rope_theta() -> f32 {
    500_000.0
}

/// `rope_parameters` block; only `rope_theta` is read (`rope_type` is
/// `default` on the published checkpoint and no other type is supported).
#[derive(Debug, Clone, Deserialize)]
pub struct MuseAssistantRopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

/// Affine quantization parameters of a quantized drafter export. The
/// published checkpoint is bf16 and carries none.
#[derive(Debug, Clone, Deserialize)]
pub struct MuseAssistantQuantization {
    pub group_size: i32,
    pub bits: i32,
}

/// Parsed drafter config.
#[derive(Debug, Clone, Deserialize)]
pub struct MuseAssistantConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// One entry per layer; empty reads as "every layer is sliding".
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub rope_parameters: Option<MuseAssistantRopeParameters>,
    /// Verify width the drafter was trained at, bonus row included.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Placeholder id filling the proposal slots of a draft block. Indexes
    /// the TARGET's embedding table, which the drafter borrows.
    pub mask_token_id: i32,
    /// Target layers whose post-layer residual streams feed `encoder.fc`,
    /// one per drafter layer.
    pub target_layer_ids: Vec<usize>,
    /// Layer count of the target this drafter was published for. Absent on
    /// the published checkpoint; checked against the bound target when
    /// present.
    #[serde(default)]
    pub num_target_layers: Option<usize>,
    /// Vocabulary of the target's head. Absent on the published checkpoint;
    /// checked against the bound target's `lm_head` width when present.
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Optional narrower runtime verify width, `min`-ed with `block_size`.
    #[serde(default)]
    pub runtime_block_size: Option<usize>,
    /// Accepted for parity with the DSpark config surface; the drafter's
    /// context cache keeps a plain `sliding_window`-row window regardless.
    #[serde(default)]
    pub draft_window_size: Option<usize>,
    #[serde(default)]
    pub quantization: Option<MuseAssistantQuantization>,
}

impl MuseAssistantConfig {
    /// Parse a drafter `config.json`, lifting a nested `dflash_config`
    /// block onto the top level first, then validating.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let mut flat = match value {
            serde_json::Value::Object(map) => map.clone(),
            _ => {
                return Err(format!(
                    "MuseAssistantConfig::from_json expected a JSON object, got: {value}"
                ));
            }
        };
        if let Some(serde_json::Value::Object(nested)) = flat.remove("dflash_config") {
            for key in [
                "mask_token_id",
                "target_layer_ids",
                "block_size",
                "runtime_block_size",
                "draft_window_size",
                "num_target_layers",
            ] {
                if let Some(v) = nested.get(key) {
                    flat.insert(key.to_string(), v.clone());
                }
            }
        }
        let config: Self = serde_json::from_value(serde_json::Value::Object(flat))
            .map_err(|e| format!("Muse Glimmer assistant config: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Structural checks that need no target in hand.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != MUSE_ASSISTANT_MODEL_TYPE {
            return Err(format!(
                "Muse Glimmer assistant config declares model_type {:?}, expected {MUSE_ASSISTANT_MODEL_TYPE:?}",
                self.model_type
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err("Muse Glimmer assistant num_hidden_layers must be non-zero".to_string());
        }
        if self.hidden_size == 0 || self.head_dim == 0 || self.intermediate_size == 0 {
            return Err(
                "Muse Glimmer assistant hidden_size, head_dim and intermediate_size must be non-zero"
                    .to_string(),
            );
        }
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err(
                "Muse Glimmer assistant num_attention_heads and num_key_value_heads must be non-zero"
                    .to_string(),
            );
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "Muse Glimmer assistant attention heads ({}) must be divisible by KV heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if !self.layer_types.is_empty() {
            if self.layer_types.len() != self.num_hidden_layers {
                return Err(format!(
                    "Muse Glimmer assistant layer_types length ({}) must equal num_hidden_layers ({})",
                    self.layer_types.len(),
                    self.num_hidden_layers
                ));
            }
            for (idx, layer_type) in self.layer_types.iter().enumerate() {
                if layer_type != SLIDING_ATTENTION {
                    return Err(format!(
                        "Muse Glimmer assistant layer {idx} has layer_type {layer_type:?}; every \
                         drafter layer must be {SLIDING_ATTENTION:?} (the bidirectional sliding \
                         mask is the only mask the drafter builds)"
                    ));
                }
            }
        }
        if self.sliding_window == 0 {
            return Err("Muse Glimmer assistant sliding_window must be non-zero".to_string());
        }
        if self.target_layer_ids.len() != self.num_hidden_layers {
            return Err(format!(
                "Muse Glimmer assistant target_layer_ids {:?} must name one target layer per \
                 drafter layer ({})",
                self.target_layer_ids, self.num_hidden_layers
            ));
        }
        if !self.target_layer_ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(format!(
                "Muse Glimmer assistant target_layer_ids {:?} must be strictly increasing",
                self.target_layer_ids
            ));
        }
        if let Some(num_target_layers) = self.num_target_layers
            && self
                .target_layer_ids
                .last()
                .is_some_and(|&last| last >= num_target_layers)
        {
            return Err(format!(
                "Muse Glimmer assistant target_layer_ids {:?} reach past num_target_layers = \
                 {num_target_layers}",
                self.target_layer_ids
            ));
        }
        if self.block_size < 2 {
            return Err(format!(
                "Muse Glimmer assistant block_size {} must be at least 2 (one bonus row plus one \
                 proposal)",
                self.block_size
            ));
        }
        if self.runtime_block_size.is_some_and(|n| n < 2) {
            return Err(format!(
                "Muse Glimmer assistant runtime_block_size {:?} must be at least 2",
                self.runtime_block_size
            ));
        }
        if self.mask_token_id < 0 {
            return Err(format!(
                "Muse Glimmer assistant mask_token_id {} must be non-negative",
                self.mask_token_id
            ));
        }
        if let Some(vocab) = self.vocab_size
            && usize::try_from(self.mask_token_id).is_ok_and(|id| id >= vocab)
        {
            return Err(format!(
                "Muse Glimmer assistant mask_token_id {} is outside the vocabulary [0, {vocab})",
                self.mask_token_id
            ));
        }
        if let Some(q) = &self.quantization {
            crate::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|err| format!("Muse Glimmer assistant {err}"))?;
        }
        Ok(())
    }

    /// RoPE base; the published checkpoint declares 500000.
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    /// Verify width the drafter runs at by default:
    /// `min(block_size, runtime_block_size)`, bonus row included.
    pub fn runtime_verify_width(&self) -> usize {
        self.runtime_block_size
            .map(|n| n.min(self.block_size))
            .unwrap_or(self.block_size)
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

/// Whether a parsed `config.json` describes a Muse Glimmer assistant
/// drafter. Either marker is enough: the `model_type` the drafter loader
/// keys on, or the `architectures` entry HuggingFace `AutoModel` keys on.
///
/// Used by: `is_dflash_drafter_config` (standalone-model rejection), the
/// `mlxcel-server` Muse Glimmer startup guard.
pub fn is_muse_assistant_config(config: &serde_json::Value) -> bool {
    if config.get("model_type").and_then(serde_json::Value::as_str)
        == Some(MUSE_ASSISTANT_MODEL_TYPE)
    {
        return true;
    }
    config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|arch| arch.as_str() == Some(MUSE_ASSISTANT_ARCHITECTURE))
        })
}

/// Whether `path/config.json` describes a Muse Glimmer assistant drafter.
/// A missing or unparseable config reads as `false`, the same convention as
/// `is_dflash_drafter_dir`: a pre-load guard leaves the loader's own error
/// to the loader.
pub fn is_muse_assistant_dir(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path.join("config.json")) else {
        return false;
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    is_muse_assistant_config(&json)
}

/// Read the default verify width of a Muse Glimmer assistant drafter at
/// `path`, `min(block_size, runtime_block_size)`, or `None` when the
/// directory is not one (or its config cannot be read).
///
/// Used by: `resolve_draft_block_size` in the `mlxcel` binary crate.
pub fn peek_muse_assistant_configured_block_size(path: &Path) -> Option<usize> {
    let bytes = std::fs::read(path.join("config.json")).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    if !is_muse_assistant_config(&json) {
        return None;
    }
    MuseAssistantConfig::from_json(&json)
        .ok()
        .map(|config| config.runtime_verify_width())
}
