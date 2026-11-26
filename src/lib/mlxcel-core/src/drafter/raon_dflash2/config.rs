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

//! [`RaonDFlash2Config`]: the `config.json` contract of the Raon-1 mini
//! DFlash2 drafter.
//!
//! `config.json` is untrusted input. Every width that sizes an allocation
//! or a reshape is bounded here, before any weight is read, because an MLX
//! shape or allocation failure crosses the cxx bridge as a process abort.

use std::path::Path;

use serde::Deserialize;

/// `architectures[0]` of the published drafter.
pub const RAON_DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";

/// Largest draft block (bonus plus proposals) the drafter serves. The vendor
/// runtime drafts 5 or 7 tokens (blocks of 6 or 8); the target's
/// row-invariant verify covers up to 16 rows.
pub const MAX_BLOCK_SIZE: usize = 16;

/// Upper bound on any single width read from the config.
const MAX_WIDTH: usize = 1 << 20;

/// Nested `dflash_config` block.
#[derive(Debug, Clone, Deserialize)]
struct DFlashBlock {
    block_size: usize,
    #[serde(default)]
    causal: bool,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: i32,
    #[serde(default = "default_true")]
    selector_enabled: bool,
    selector_rank: usize,
    selector_top_k: usize,
    swa_window_size: usize,
    target_feature_size: usize,
    target_layer_ids: Vec<usize>,
    #[serde(default = "default_streams")]
    target_streams_per_layer: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
}

/// Raw top-level fields, before validation.
#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    vocab_size: usize,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    dflash_config: DFlashBlock,
}

fn default_true() -> bool {
    true
}
fn default_streams() -> usize {
    4
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}

/// Validated DFlash2 drafter configuration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RaonDFlash2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    /// Trained block width (bonus plus `block_size - 1` masks).
    pub block_size: usize,
    pub mask_token_id: i32,
    /// Context positions each layer keeps (`swa_window_size`).
    pub window: usize,
    /// Taps of the grouped convolution (`conv_kernel_size`).
    pub conv_taps: usize,
    /// Channels per convolution group (`conv_group_size`).
    pub conv_group_size: usize,
    pub selector_top_k: usize,
    pub selector_rank: usize,
    /// Zero-based target decoder layers whose output streams feed `fc`.
    pub target_layer_ids: Vec<usize>,
    /// mHC streams captured per target layer.
    pub target_streams_per_layer: usize,
    /// `len(target_layer_ids) * target_streams_per_layer * hidden_size`.
    pub target_feature_size: usize,
}

impl RaonDFlash2Config {
    /// Whether a raw `config.json` declares this drafter.
    pub fn is_raon_dflash2_config(config: &serde_json::Value) -> bool {
        config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|archs| {
                archs
                    .iter()
                    .any(|a| a.as_str() == Some(RAON_DFLASH2_ARCHITECTURE))
            })
    }

    /// Whether `path/config.json` declares this drafter.
    pub fn is_raon_dflash2_dir(path: &Path) -> bool {
        std::fs::read(path.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .is_some_and(|config| Self::is_raon_dflash2_config(&config))
    }

    /// Parse and validate.
    ///
    /// # Errors
    /// Returns a message naming the offending field.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let raw: RawConfig = serde_json::from_value(value.clone())
            .map_err(|e| format!("Raon DFlash2 config.json: {e}"))?;
        if !raw
            .architectures
            .iter()
            .any(|a| a == RAON_DFLASH2_ARCHITECTURE)
        {
            return Err(format!(
                "Raon DFlash2 config.json: architectures {:?} does not name \
                 {RAON_DFLASH2_ARCHITECTURE}",
                raw.architectures
            ));
        }
        let d = raw.dflash_config;
        let rope_theta = raw
            .rope_parameters
            .map(|r| r.rope_theta)
            .or(raw.rope_theta)
            .ok_or("Raon DFlash2 config.json: rope_parameters.rope_theta is missing")?;
        let config = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: rope_theta as f32,
            vocab_size: raw.vocab_size,
            block_size: d.block_size,
            mask_token_id: d.mask_token_id,
            window: d.swa_window_size,
            conv_taps: d.conv_kernel_size,
            conv_group_size: d.conv_group_size,
            selector_top_k: d.selector_top_k,
            selector_rank: d.selector_rank,
            target_layer_ids: d.target_layer_ids,
            target_streams_per_layer: d.target_streams_per_layer,
            target_feature_size: d.target_feature_size,
        };
        if d.causal {
            return Err(
                "Raon DFlash2 config.json: dflash_config.causal is true; the \
                        DFlash2 drafter attends its block non-causally"
                    .to_string(),
            );
        }
        if !d.selector_enabled {
            return Err(
                "Raon DFlash2 config.json: dflash_config.selector_enabled is \
                        false; only the candidate-selector head is implemented"
                    .to_string(),
            );
        }
        config.validate()?;
        Ok(config)
    }

    /// Bounds and cross-field consistency.
    ///
    /// # Errors
    /// Returns a message naming the offending field.
    pub fn validate(&self) -> Result<(), String> {
        let widths = [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("dflash_config.swa_window_size", self.window),
            ("dflash_config.conv_kernel_size", self.conv_taps),
            ("dflash_config.conv_group_size", self.conv_group_size),
            ("dflash_config.selector_top_k", self.selector_top_k),
            ("dflash_config.selector_rank", self.selector_rank),
            (
                "dflash_config.target_streams_per_layer",
                self.target_streams_per_layer,
            ),
        ];
        for (name, value) in widths {
            if value == 0 || value > MAX_WIDTH {
                return Err(format!(
                    "Raon DFlash2 config.json: {name} = {value} is outside 1..={MAX_WIDTH}"
                ));
            }
        }
        if self.vocab_size == 0 || self.vocab_size > i32::MAX as usize {
            return Err(format!(
                "Raon DFlash2 config.json: vocab_size = {} is outside 1..=i32::MAX",
                self.vocab_size
            ));
        }
        if self.mask_token_id < 0 || self.mask_token_id as usize >= self.vocab_size {
            return Err(format!(
                "Raon DFlash2 config.json: dflash_config.mask_token_id {} is outside the \
                 vocabulary (vocab_size {})",
                self.mask_token_id, self.vocab_size
            ));
        }
        if !(2..=MAX_BLOCK_SIZE).contains(&self.block_size) {
            return Err(format!(
                "Raon DFlash2 config.json: dflash_config.block_size {} is outside \
                 2..={MAX_BLOCK_SIZE}",
                self.block_size
            ));
        }
        if self.hidden_size % self.conv_group_size != 0 {
            return Err(format!(
                "Raon DFlash2 config.json: hidden_size {} is not a multiple of \
                 dflash_config.conv_group_size {}",
                self.hidden_size, self.conv_group_size
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(format!(
                "Raon DFlash2 config.json: num_attention_heads {} is not a multiple of \
                 num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.selector_top_k > self.vocab_size {
            return Err(format!(
                "Raon DFlash2 config.json: dflash_config.selector_top_k {} exceeds \
                 vocab_size {}",
                self.selector_top_k, self.vocab_size
            ));
        }
        if self.window < MAX_BLOCK_SIZE {
            return Err(format!(
                "Raon DFlash2 config.json: dflash_config.swa_window_size {} is smaller \
                 than the widest block ({MAX_BLOCK_SIZE})",
                self.window
            ));
        }
        self.validate_target_layer_ids()
    }

    /// `target_layer_ids` must be distinct, ascending, and account for
    /// exactly `target_feature_size` features.
    fn validate_target_layer_ids(&self) -> Result<(), String> {
        let ids = &self.target_layer_ids;
        if ids.is_empty() || ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!(
                "Raon DFlash2 config.json: dflash_config.target_layer_ids {ids:?} must be a \
                 non-empty, strictly ascending list"
            ));
        }
        let expected = ids
            .len()
            .checked_mul(self.target_streams_per_layer)
            .and_then(|n| n.checked_mul(self.hidden_size));
        if expected != Some(self.target_feature_size) {
            return Err(format!(
                "Raon DFlash2 config.json: dflash_config.target_layer_ids {ids:?} x \
                 target_streams_per_layer {} x hidden_size {} does not equal \
                 target_feature_size {}",
                self.target_streams_per_layer, self.hidden_size, self.target_feature_size
            ));
        }
        Ok(())
    }

    /// Whether the target can feed this drafter: every tap layer exists, the
    /// target carries `target_streams_per_layer` residual streams of
    /// `hidden_size` features, and the vocabularies agree.
    ///
    /// # Errors
    /// Returns a message naming the mismatch (`target_layer_ids`,
    /// `target_streams_per_layer`, `hidden_size` or `vocab_size`).
    pub fn validate_target(
        &self,
        target_layers: usize,
        target_streams: Option<usize>,
        target_hidden: usize,
        target_vocab: usize,
    ) -> Result<(), String> {
        if let Some(&last) = self.target_layer_ids.last()
            && last >= target_layers
        {
            return Err(format!(
                "Raon DFlash2 drafter target_layer_ids {:?} names layer {last}, but the \
                 target has {target_layers} decoder layers",
                self.target_layer_ids
            ));
        }
        if target_streams != Some(self.target_streams_per_layer) {
            return Err(format!(
                "Raon DFlash2 drafter reads target_streams_per_layer = {} residual streams \
                 per tap layer, but the target carries {}",
                self.target_streams_per_layer,
                target_streams.map_or_else(|| "none".to_string(), |n| n.to_string())
            ));
        }
        if target_hidden != self.hidden_size {
            return Err(format!(
                "Raon DFlash2 drafter hidden_size {} differs from the target's {target_hidden}",
                self.hidden_size
            ));
        }
        if target_vocab != self.vocab_size {
            return Err(format!(
                "Raon DFlash2 drafter vocab_size {} differs from the target's {target_vocab}",
                self.vocab_size
            ));
        }
        Ok(())
    }

    /// Convolution groups per row (`hidden_size / conv_group_size`).
    pub fn conv_groups(&self) -> usize {
        self.hidden_size / self.conv_group_size
    }
}
