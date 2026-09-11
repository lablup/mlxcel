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

//! Config for the Cohere Compass text decoder (`cohere_compass_text`).
//!
//! Split out of `cohere_compass.rs` so the decoder file stays inside the
//! project's per-file size budget. The interesting part is
//! [`CompassTextConfig::rope_for_layer_type`]: Compass keys RoPE off the layer
//! type, and a `null` entry means the layer type gets no positional encoding at
//! all, not "fall back to the default RoPE".

use serde::Deserialize;
use serde_json::Value;

/// `layer_types` entry for a layer that attends the whole prefix.
pub const FULL_ATTENTION: &str = "full_attention";
/// `layer_types` entry for a layer restricted to `sliding_window` keys.
pub const SLIDING_ATTENTION: &str = "sliding_attention";

/// Resolved RoPE for one layer type.
///
/// Every rotated layer in this family uses the 3-axis MRoPE; `mrope_section`
/// is `[h, w, t]` and falls back to upstream's
/// [`crate::models::cohere_compass_rope::DEFAULT_MROPE_SECTION`] when the entry
/// omits it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRopeSpec {
    pub theta: f32,
    pub mrope_section: [i32; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenIds {
    Single(i32),
    Multiple(Vec<i32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompassQuantConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

/// `text_config` of a `cohere_compass` checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct CompassTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// `"layer_norm"` (the published checkpoints) or `"rms_norm"`.
    #[serde(default = "default_norm_type")]
    pub norm_type: String,
    /// `"parallel"` (the published checkpoints) or `"sequential"`.
    #[serde(default = "default_block_type")]
    pub transformer_block_type: String,
    #[serde(default = "default_logit_scale")]
    pub logit_scale: f32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Per-layer-type RoPE. See [`Self::rope_for_layer_type`].
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Accepted for forward compatibility; only `"split"` is implemented,
    /// which is what upstream does unconditionally.
    #[serde(default = "default_rope_style")]
    pub rope_style: String,
    #[serde(default = "default_true")]
    pub rope_on_all_layers: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenIds>,
    #[serde(default)]
    pub quantization: Option<CompassQuantConfig>,
}

fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_norm_type() -> String {
    "layer_norm".to_string()
}
fn default_block_type() -> String {
    "parallel".to_string()
}
fn default_logit_scale() -> f32 {
    1.0
}
fn default_sliding_window() -> usize {
    4096
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rope_style() -> String {
    "split".to_string()
}
fn default_true() -> bool {
    true
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl CompassTextConfig {
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
            .unwrap_or(0)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(0)
    }

    /// Upstream rotates with `rotate_half` over the two halves of `head_dim`
    /// and offers no alternative, so anything but `"split"` is refused rather
    /// than silently rotated the wrong way.
    pub fn validate_rope_style(&self) -> Result<(), String> {
        if self.rope_style == "split" {
            return Ok(());
        }
        Err(format!(
            "cohere_compass: rope_style {:?} is not implemented; the reference implementation \
             always rotates the two halves of head_dim (\"split\")",
            self.rope_style
        ))
    }

    /// One `layer_types` entry per decoder layer.
    ///
    /// An absent list means every layer attends the full prefix, which is the
    /// transformers default when `layer_types` is omitted.
    pub fn resolved_layer_types(&self) -> Result<Vec<String>, String> {
        if self.layer_types.is_empty() {
            return Ok(vec![FULL_ATTENTION.to_string(); self.num_hidden_layers]);
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "cohere_compass: layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for entry in &self.layer_types {
            if entry != FULL_ATTENTION && entry != SLIDING_ATTENTION {
                return Err(format!(
                    "cohere_compass: unknown layer_types entry {entry:?} (expected \
                     {FULL_ATTENTION:?} or {SLIDING_ATTENTION:?})"
                ));
            }
        }
        Ok(self.layer_types.clone())
    }

    /// Resolve the RoPE for one layer type.
    ///
    /// `Ok(None)` means the layer type carries no positional encoding at all.
    /// That is the whole point of the block: the published checkpoint sets
    /// `rope_parameters.full_attention = null`, and reading `null` as "use the
    /// default" would silently rotate the global layers that the model trained
    /// without any rotation.
    ///
    /// `rope_on_all_layers` is only consulted when `rope_parameters` carries no
    /// per-layer-type entries at all.
    pub fn rope_for_layer_type(&self, layer_type: &str) -> Result<Option<LayerRopeSpec>, String> {
        let params = self.rope_parameters.as_ref().and_then(|v| v.as_object());
        let fallback_theta = params
            .and_then(|m| m.get("rope_theta"))
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(self.rope_theta);

        // A per-layer-type entry is a key whose value is an object (a spec) or
        // `null` (explicitly no RoPE). The sibling scalars `rope_theta` and
        // `rope_type` are the block-level defaults, not layer types.
        let has_per_layer_entries =
            params.is_some_and(|m| m.values().any(|v| v.is_null() || v.is_object()));

        if has_per_layer_entries {
            return match params.and_then(|m| m.get(layer_type)) {
                None => Ok(None),
                Some(v) if v.is_null() => Ok(None),
                Some(v) => parse_rope_entry(v, fallback_theta).map(Some),
            };
        }

        if !self.rope_on_all_layers {
            return Ok(None);
        }
        match params {
            Some(m) => parse_rope_entry(&Value::Object(m.clone()), fallback_theta).map(Some),
            None => Ok(Some(LayerRopeSpec {
                theta: fallback_theta,
                mrope_section: crate::models::cohere_compass_rope::DEFAULT_MROPE_SECTION,
            })),
        }
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(EosTokenIds::Single(id)) => vec![*id],
            Some(EosTokenIds::Multiple(ids)) if !ids.is_empty() => ids.clone(),
            // `<|END_OF_TURN_TOKEN|>` on every published Cohere checkpoint.
            _ => vec![255001],
        }
    }
}

fn parse_rope_entry(value: &Value, fallback_theta: f32) -> Result<LayerRopeSpec, String> {
    let theta = value
        .get("rope_theta")
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(fallback_theta);

    let mrope_section = match value.get("mrope_section") {
        None | Some(Value::Null) => crate::models::cohere_compass_rope::DEFAULT_MROPE_SECTION,
        Some(section) => {
            let arr = section
                .as_array()
                .ok_or_else(|| "cohere_compass: mrope_section must be an array".to_string())?;
            if arr.len() != 3 {
                return Err(format!(
                    "cohere_compass: mrope_section must have 3 entries (H, W, T), got {}",
                    arr.len()
                ));
            }
            let mut out = [0i32; 3];
            for (i, entry) in arr.iter().enumerate() {
                out[i] = entry
                    .as_i64()
                    .and_then(|v| i32::try_from(v).ok())
                    .ok_or_else(|| {
                        format!("cohere_compass: mrope_section[{i}] is not a 32-bit integer")
                    })?;
            }
            out
        }
    };

    Ok(LayerRopeSpec {
        theta,
        mrope_section,
    })
}
