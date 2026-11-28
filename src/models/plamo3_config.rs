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

//! PLaMo 3 `config.json` (`Plamo3Config` in the checkpoint's
//! `modeling_plamo.py`).
//!
//! Defaults follow the HF reference rather than mlx-lm's dataclass where the
//! two differ (`window_size` 128, `rope_theta` 150000), because the HF config
//! class is what the checkpoint was exported with. The legacy spellings the
//! reference still accepts (`rope_global_theta`, and `sliding_window` as an int
//! or a per-layer list) are honored the same way.

use serde::Deserialize;

use crate::models::afmoe::{Quantization, TokenIdField};
use crate::models::rope_utils::RopeScalingSpec;

const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
const MAX_HIDDEN_SIZE: usize = 65_536;
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
const MAX_HEAD_DIM: usize = 8192;
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
const MAX_VOCAB_SIZE: usize = 1 << 24;
const MAX_POSITIONS: usize = 1 << 24;

/// YaRN correction band the reference hardcodes for the full-attention layers.
const YARN_BETA_FAST: f32 = 32.0;
const YARN_BETA_SLOW: f32 = 1.0;

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_true() -> bool {
    true
}
fn default_head_dim() -> usize {
    128
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_window_size() -> usize {
    128
}
fn default_sliding_window_pattern() -> usize {
    8
}
fn default_rope_theta() -> f32 {
    150_000.0
}
fn default_rope_local_theta() -> f32 {
    10_000.0
}
fn default_rope_scaling_factor() -> f32 {
    1.0
}
fn default_linear_type() -> String {
    "normal".to_string()
}

/// PLaMo 3 `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub scale_embedding: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    /// Legacy spelling of `window_size`: an int, or a per-layer list whose
    /// non-null entries all agree. Read through [`Self::sliding_window`].
    #[serde(default)]
    pub sliding_window: Option<serde_json::Value>,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Legacy name of `rope_theta`, used only while `rope_theta` is the default.
    #[serde(default)]
    pub rope_global_theta: Option<f32>,
    #[serde(default = "default_rope_local_theta")]
    pub rope_local_theta: f32,
    /// YaRN factor for the full-attention layers; `1` means plain RoPE.
    #[serde(default = "default_rope_scaling_factor")]
    pub rope_scaling_factor: f32,
    /// YaRN original context, required when `rope_scaling_factor != 1`.
    #[serde(default)]
    pub initial_context_length: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub image_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub image_feature_size: Option<serde_json::Value>,
    #[serde(default = "default_linear_type")]
    pub linear_type: String,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// Whether layer `layer_idx` is a full-attention layer:
/// `(layer_idx + 1) % pattern == 0`, so pattern 8 makes layers 7, 15, ... full.
///
/// `pattern` is validated non-zero before any layer is built.
pub fn is_full_attention(sliding_window_pattern: usize, layer_idx: usize) -> bool {
    (layer_idx + 1).is_multiple_of(sliding_window_pattern.max(1))
}

impl ModelArgs {
    /// The sliding window in tokens: `sliding_window` when the config carries
    /// a usable legacy value, `window_size` otherwise (HF `Plamo3Config`).
    pub fn sliding_window(&self) -> usize {
        match &self.sliding_window {
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(self.window_size),
            Some(serde_json::Value::Array(items)) => {
                let windows: Vec<u64> = items.iter().filter_map(|v| v.as_u64()).collect();
                match windows.first() {
                    Some(&first) if windows.iter().all(|&w| w == first) => {
                        usize::try_from(first).unwrap_or(self.window_size)
                    }
                    _ => self.window_size,
                }
            }
            _ => self.window_size,
        }
    }

    /// The full-attention base, after the legacy `rope_global_theta` alias.
    pub fn full_rope_theta(&self) -> f32 {
        match self.rope_global_theta {
            Some(legacy) if self.rope_theta == default_rope_theta() => legacy,
            _ => self.rope_theta,
        }
    }

    pub fn is_full_layer(&self, layer_idx: usize) -> bool {
        is_full_attention(self.sliding_window_pattern, layer_idx)
    }

    /// First full-attention layer, whose cache anchors the causal mask.
    pub fn full_attention_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| self.is_full_layer(i))
    }

    /// First sliding layer, whose cache anchors the windowed mask. `None`
    /// when `num_hidden_layers < sliding_window_pattern` leaves none.
    pub fn sliding_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| !self.is_full_layer(i))
    }

    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(TokenIdField::Single(id)) => vec![*id],
            Some(TokenIdField::Multiple(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }

    /// The YaRN block the reference's `rope_parameters["full_attention"]`
    /// builds when `rope_scaling_factor != 1`: `beta_fast` 32, `beta_slow` 1,
    /// `original_max_position_embeddings = initial_context_length` and
    /// `truncate: false` (the correction band keeps its fractional bounds).
    /// `None` for plain RoPE. Sliding layers never scale.
    pub fn full_attention_rope_spec(&self) -> Option<RopeScalingSpec> {
        if self.rope_scaling_factor == 1.0 {
            return None;
        }
        Some(RopeScalingSpec {
            rope_type: Some("yarn".to_string()),
            factor: Some(self.rope_scaling_factor),
            original_max_position_embeddings: self.initial_context_length.map(|n| n as f32),
            beta_fast: Some(YARN_BETA_FAST),
            beta_slow: Some(YARN_BETA_SLOW),
            truncate: Some(false),
            ..RopeScalingSpec::default()
        })
    }

    /// Reject a config that cannot describe a real PLaMo 3 before any field
    /// sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        let bounded = |name: &str, value: usize, max: usize| {
            if value == 0 || value > max {
                Err(format!(
                    "PLaMo 3 {name} ({value}) must be between 1 and {max}"
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
        bounded(
            "intermediate_size",
            self.intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        bounded("vocab_size", self.vocab_size, MAX_VOCAB_SIZE)?;
        bounded(
            "max_position_embeddings",
            self.max_position_embeddings,
            MAX_POSITIONS,
        )?;
        bounded(
            "sliding_window_pattern",
            self.sliding_window_pattern,
            MAX_NUM_HIDDEN_LAYERS,
        )?;
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "PLaMo 3 num_key_value_heads ({}) must divide num_attention_heads ({})",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "PLaMo 3 head_dim ({}) must be even; RoPE rotates channel pairs",
                self.head_dim
            ));
        }
        crate::models::model_owned::cache_window(
            "PLaMo 3",
            self.sliding_window(),
            self.sliding_index().is_some(),
        )?;
        if self.scale_embedding {
            return Err(
                "PLaMo 3 scale_embedding = true is not supported: every public checkpoint sets \
                 it false, and this loader does not scale the embeddings by sqrt(hidden_size)"
                    .to_string(),
            );
        }
        for (name, value) in [
            ("image_token_id", &self.image_token_id),
            ("image_feature_size", &self.image_feature_size),
        ] {
            if value.as_ref().is_some_and(|v| !v.is_null()) {
                return Err(format!(
                    "PLaMo 3 {name} is set, but the image projection is not supported; only \
                     the text-only checkpoints (where it is null) load"
                ));
            }
        }
        if self.linear_type != "normal" {
            return Err(format!(
                "PLaMo 3 linear_type {:?} is not supported; only \"normal\" (unquantized or \
                 MLX-quantized linears) loads",
                self.linear_type
            ));
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.full_rope_theta()),
            ("rope_local_theta", self.rope_local_theta),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "PLaMo 3 {name} ({value}) must be a finite positive number"
                ));
            }
        }
        self.validate_yarn()?;
        if let Some(q) = self.quantization.as_ref() {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("PLaMo 3 config.json: {e}"))?;
        }
        Ok(())
    }

    /// The reference asserts `initial_context_length` is set whenever the
    /// factor is not 1, and that the YaRN band satisfies
    /// `0 < low < high < head_dim / 2 - 1`; a config that breaks either would
    /// fail to load there, so it fails here too instead of building a clamped
    /// table the checkpoint was never trained with.
    fn validate_yarn(&self) -> Result<(), String> {
        let factor = self.rope_scaling_factor;
        if factor == 1.0 {
            return Ok(());
        }
        if !factor.is_finite() || factor < 1.0 {
            return Err(format!(
                "PLaMo 3 rope_scaling_factor ({factor}) must be a finite number >= 1"
            ));
        }
        let Some(original) = self.initial_context_length.filter(|&n| n > 0) else {
            return Err(format!(
                "PLaMo 3 rope_scaling_factor is {factor} but initial_context_length is unset; \
                 the YaRN full-attention rotation needs the original context"
            ));
        };
        let d_h = (self.head_dim / 2) as f64;
        let base = f64::from(self.full_rope_theta()).ln();
        let band = |beta: f32| {
            d_h * (original as f64 / (f64::from(beta) * 2.0 * std::f64::consts::PI)).ln() / base
        };
        let (low, high) = (band(YARN_BETA_FAST), band(YARN_BETA_SLOW));
        if !(0.0 < low && low < high && high < d_h - 1.0) {
            return Err(format!(
                "PLaMo 3 YaRN band [{low}, {high}] is outside (0, {}) for head_dim {}, \
                 rope_theta {} and initial_context_length {original}",
                d_h - 1.0,
                self.head_dim,
                self.full_rope_theta()
            ));
        }
        Ok(())
    }
}
