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

//! Configuration for the Gemma 4 MTP "assistant" drafter.
//!
//! Mirrors the upstream `Gemma4AssistantConfig` (HF-compatible, flattened) and
//! its nested `TextConfig`. See
//! `references/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/config.py`
//! and `references/mlx-vlm/mlx_vlm/models/gemma4/config.py`.
//!
//! The drafter `TextConfig` is intentionally a self-contained subset that lives
//! inside `mlxcel-core`. The full Gemma 4 target `TextConfig` lives in
//! `crate::models::gemma4::TextConfig` (the `mlxcel` crate), which sits above
//! `mlxcel-core` in the dependency graph and therefore cannot be imported here.
//! Both definitions deserialize the same HF schema; field names and defaults
//! match upstream Python.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// RoPE parameters per layer-type (mirrors
/// `crate::models::gemma4::RopeParameters`).
///
/// Each Gemma 4 layer is tagged with `"full_attention"` or
/// `"sliding_attention"`, and each tag has its own RoPE configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DrafterRopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// `"proportional"` for Gemma 4 full-attention layers (exponents
    /// normalized by the full `head_dim`), `"default"` for sliding-attention
    /// (standard `nn.RoPE(dims = head_dim * partial_rotary_factor)` path).
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

/// Quantization arguments for the drafter checkpoint.
///
/// Mirrors `crate::models::gemma4::QuantizationArgs`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DrafterQuantizationArgs {
    pub group_size: usize,
    pub bits: usize,
}

/// Text-side configuration for the Gemma 4 assistant drafter (the inner
/// `text_config` of `Gemma4AssistantConfig`).
///
/// Mirrors upstream `mlx_vlm.models.gemma4.config.TextConfig`. The set of
/// fields the drafter actually exercises is a strict subset of what the full
/// Gemma 4 target uses (no MoE, no per-layer input gating, K/V always shared);
/// fields outside that subset are kept for HF parity so that a drafter
/// `config.json` produced by upstream tools loads unchanged.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DrafterTextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    /// On the drafter every layer is KV-shared. Upstream `__post_init__` sets
    /// this to `num_hidden_layers` when missing/zero (see
    /// `Gemma4AssistantConfig.__post_init__`). The mlxcel port mirrors that
    /// fixup inside [`Gemma4AssistantConfig::normalize`].
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    pub rope_parameters: HashMap<String, DrafterRopeParameters>,
    pub sliding_window: usize,
    #[serde(default)]
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub quantization: Option<DrafterQuantizationArgs>,
}

impl DrafterTextConfig {
    /// Effective group size used for quantized weight loading. Mirrors
    /// `crate::models::gemma4::TextConfig::group_size`.
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64)
    }

    /// Effective bit width for quantized weights. Mirrors
    /// `crate::models::gemma4::TextConfig::bits`.
    pub fn bits(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4)
    }

    /// Layer-type string (`"full_attention"` or `"sliding_attention"`) for the
    /// given layer index. Panics on out-of-range index — the caller must
    /// validate against `num_hidden_layers` first.
    pub fn layer_type(&self, layer_idx: usize) -> &str {
        self.layer_types[layer_idx].as_str()
    }

    /// Returns `true` when the layer has the sliding-window-attention layer
    /// type.
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == "sliding_attention"
    }

    /// Effective head_dim for the given layer. Full-attention layers use
    /// `global_head_dim` (when present), sliding-attention layers always use
    /// `head_dim`. Mirrors `crate::models::gemma4::TextConfig::head_dim_for_layer`.
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> i32 {
        if self.is_sliding_layer(layer_idx) {
            self.head_dim as i32
        } else {
            self.global_head_dim.unwrap_or(self.head_dim) as i32
        }
    }

    /// Effective num_kv_heads for the given layer. Full-attention layers with
    /// `attention_k_eq_v = true` collapse to `num_global_key_value_heads`
    /// (when present), otherwise fall back to `num_key_value_heads`. Mirrors
    /// `crate::models::gemma4::TextConfig::num_kv_heads_for_layer`.
    pub fn num_kv_heads_for_layer(&self, layer_idx: usize) -> i32 {
        if self.attention_k_eq_v && !self.is_sliding_layer(layer_idx) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads) as i32
        } else {
            self.num_key_value_heads as i32
        }
    }

    /// RoPE parameters for the given layer's layer-type key (one of
    /// `"full_attention"` / `"sliding_attention"`). Falls back to default
    /// values when the key is missing.
    pub fn rope_params_for_layer(&self, layer_idx: usize) -> DrafterRopeParameters {
        let key = if self.is_sliding_layer(layer_idx) {
            "sliding_attention"
        } else {
            "full_attention"
        };
        self.rope_parameters
            .get(key)
            .cloned()
            .unwrap_or(DrafterRopeParameters {
                rope_theta: default_rope_theta(),
                partial_rotary_factor: default_partial_rotary_factor(),
                rope_type: default_rope_type(),
            })
    }
}

/// Drafter config for Gemma 4 Multi-Token Prediction (assistant) models.
///
/// Mirrors the HF `Gemma4AssistantConfig` shape: top-level drafter knobs +
/// nested `text_config`. Defaults match the upstream Python defaults for
/// `gg-hf-am/gemma-4-26B-A4B-it-assistant`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Gemma4AssistantConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_backbone_hidden_size")]
    pub backbone_hidden_size: usize,
    /// `true` for the E2B / E4B drafters (centroid-routed sparse LM head); the
    /// 26B-A4B / 31B drafters keep this `false` and tie weights to the
    /// drafter's `embed_tokens`.
    #[serde(default)]
    pub use_ordered_embeddings: bool,
    /// Number of centroids the `MaskedEmbedder` scores in the sparse path.
    /// Default 2048 matches the upstream `Gemma4AssistantConfig.num_centroids`.
    #[serde(default = "default_num_centroids")]
    pub num_centroids: usize,
    /// Top-K clusters materialised at each step in the sparse path. Default 32
    /// matches the upstream `centroid_intermediate_top_k`.
    #[serde(default = "default_centroid_top_k")]
    pub centroid_intermediate_top_k: usize,
    /// `true` when the drafter's LM head shares weights with its
    /// `embed_tokens`. On all four supported drafters the canonical config
    /// keeps this `true`.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    /// Number of speculatively drafted tokens per round (the first token is
    /// the most recently accepted bonus, so the drafter actually emits
    /// `block_size - 1` candidates).
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Unused by MTP — kept for API parity with the DFlash round-loop so
    /// `draft_model.config.target_layer_ids` lookups do not crash.
    #[serde(default)]
    pub target_layer_ids: Vec<usize>,
    /// Captured target layer types (`"full_attention"` / `"sliding_attention"`
    /// per layer). Filled in by `bind()` when the target's `language_model.
    /// config.layer_types` is available; left empty otherwise.
    #[serde(default)]
    pub target_layer_types: Vec<String>,
    /// Nested text config. Required at construction time; the
    /// [`Gemma4AssistantConfig::normalize`] entry point validates this and
    /// fixes up `num_kv_shared_layers` to mirror upstream `__post_init__`.
    pub text_config: Option<DrafterTextConfig>,
}

fn default_model_type() -> String {
    "gemma4_assistant".to_string()
}

fn default_backbone_hidden_size() -> usize {
    1536
}

fn default_num_centroids() -> usize {
    2048
}

fn default_centroid_top_k() -> usize {
    32
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_block_size() -> usize {
    4
}

impl Gemma4AssistantConfig {
    /// Apply the upstream `__post_init__` fixup: when `num_kv_shared_layers`
    /// is missing or zero, set it to `num_hidden_layers` so every drafter
    /// layer reads from shared K/V. Returns `Err` when `text_config` is
    /// absent (the upstream Python raises `ValueError` in that case).
    pub fn normalize(mut self) -> Result<Self, String> {
        let text_cfg = self
            .text_config
            .as_mut()
            .ok_or_else(|| "Gemma4AssistantConfig.text_config must be set".to_string())?;
        if text_cfg.num_kv_shared_layers == 0 {
            text_cfg.num_kv_shared_layers = text_cfg.num_hidden_layers;
        }
        Ok(self)
    }

    /// Convenience accessor: returns the nested text config or panics with a
    /// load-time message. Use after [`Self::normalize`] has succeeded.
    pub fn text_config(&self) -> &DrafterTextConfig {
        self.text_config
            .as_ref()
            .expect("Gemma4AssistantConfig.text_config must be set (call normalize first)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HF-style drafter config (E4B variant — `use_ordered_embeddings=true`
    /// with `tie_word_embeddings=true`, centroid LM head).
    #[test]
    fn deserialize_e4b_style_config_round_trips_defaults() {
        // text_config bare-minimum: matches the upstream Python defaults that
        // the assistant `config.json` ships with.
        let json = r#"{
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": true,
            "num_centroids": 2048,
            "centroid_intermediate_top_k": 32,
            "tie_word_embeddings": true,
            "block_size": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 1024,
                "num_attention_heads": 4,
                "head_dim": 64,
                "rms_norm_eps": 1.0e-6,
                "vocab_size": 262144,
                "num_key_value_heads": 1,
                "rope_parameters": {
                    "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 1.0, "rope_type": "proportional"},
                    "sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0, "rope_type": "default"}
                },
                "sliding_window": 512,
                "sliding_window_pattern": 5,
                "max_position_embeddings": 131072,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
            }
        }"#;
        let cfg: Gemma4AssistantConfig = serde_json::from_str(json).expect("parse config");
        let cfg = cfg.normalize().expect("normalize");

        assert_eq!(cfg.model_type, "gemma4_assistant");
        assert_eq!(cfg.backbone_hidden_size, 256);
        assert!(cfg.use_ordered_embeddings);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.block_size, 4);
        let tc = cfg.text_config();
        assert_eq!(tc.num_hidden_layers, 4);
        // num_kv_shared_layers was 0 in JSON; normalize() must promote it to
        // num_hidden_layers per upstream __post_init__.
        assert_eq!(tc.num_kv_shared_layers, 4);
        // RoPE params resolve per layer.
        let last_layer = tc.num_hidden_layers - 1;
        let rope = tc.rope_params_for_layer(last_layer);
        assert_eq!(rope.rope_type, "proportional");
    }

    /// The Gemma 4 Unified 12B drafter ships `model_type:
    /// gemma4_unified_assistant` with `text_config.model_type:
    /// gemma4_unified_text` and `backbone_hidden_size: 3840`. The config must
    /// parse and normalize exactly like the non-unified assistant spelling —
    /// neither `model_type` is hard-rejected (both are free-form strings).
    #[test]
    fn deserialize_gemma4_unified_assistant_config_parses_and_normalizes() {
        let json = r#"{
            "model_type": "gemma4_unified_assistant",
            "backbone_hidden_size": 3840,
            "use_ordered_embeddings": false,
            "num_centroids": 2048,
            "centroid_intermediate_top_k": 32,
            "tie_word_embeddings": true,
            "block_size": 4,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 2048,
                "num_attention_heads": 4,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1.0e-6,
                "vocab_size": 262144,
                "num_key_value_heads": 1,
                "rope_parameters": {
                    "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 1.0, "rope_type": "proportional"},
                    "sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0, "rope_type": "default"}
                },
                "sliding_window": 1024,
                "sliding_window_pattern": 4,
                "max_position_embeddings": 131072,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
            }
        }"#;
        let cfg: Gemma4AssistantConfig = serde_json::from_str(json).expect("parse unified config");
        let cfg = cfg.normalize().expect("normalize");

        assert_eq!(cfg.model_type, "gemma4_unified_assistant");
        assert_eq!(cfg.backbone_hidden_size, 3840);
        let tc = cfg.text_config();
        assert_eq!(tc.model_type, "gemma4_unified_text");
        assert_eq!(tc.hidden_size, 1024);
        assert_eq!(tc.num_hidden_layers, 4);
        // num_kv_shared_layers was omitted (0) in JSON; normalize() promotes it
        // to num_hidden_layers per upstream __post_init__.
        assert_eq!(tc.num_kv_shared_layers, 4);
        // The full-attention head uses global_head_dim when present.
        assert_eq!(tc.head_dim_for_layer(0), 256);
        assert_eq!(tc.head_dim_for_layer(3), 512);
    }

    #[test]
    fn normalize_rejects_missing_text_config() {
        let cfg = Gemma4AssistantConfig {
            model_type: "gemma4_assistant".into(),
            backbone_hidden_size: 256,
            use_ordered_embeddings: false,
            num_centroids: 2048,
            centroid_intermediate_top_k: 32,
            tie_word_embeddings: true,
            block_size: 4,
            target_layer_ids: vec![],
            target_layer_types: vec![],
            text_config: None,
        };
        let err = cfg.normalize().expect_err("must error");
        assert!(err.contains("text_config"));
    }

    #[test]
    fn layer_type_helpers_match_layer_types_array() {
        // 4-layer pattern: SWA, SWA, SWA, full — mirrors a small E-series drafter.
        let mut rope_parameters = HashMap::new();
        rope_parameters.insert(
            "full_attention".to_string(),
            DrafterRopeParameters {
                rope_theta: 1_000_000.0,
                partial_rotary_factor: 1.0,
                rope_type: "proportional".to_string(),
            },
        );
        rope_parameters.insert(
            "sliding_attention".to_string(),
            DrafterRopeParameters {
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rope_type: "default".to_string(),
            },
        );
        let tc = DrafterTextConfig {
            model_type: "gemma4_text".into(),
            hidden_size: 256,
            num_hidden_layers: 4,
            intermediate_size: 1024,
            num_attention_heads: 4,
            head_dim: 64,
            global_head_dim: Some(128),
            rms_norm_eps: 1e-6,
            vocab_size: 262144,
            num_key_value_heads: 1,
            num_global_key_value_heads: None,
            num_kv_shared_layers: 4,
            rope_parameters,
            sliding_window: 512,
            sliding_window_pattern: 5,
            max_position_embeddings: 131_072,
            layer_types: vec![
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "full_attention".into(),
            ],
            attention_k_eq_v: false,
            final_logit_softcapping: None,
            use_double_wide_mlp: false,
            quantization: None,
        };
        assert!(tc.is_sliding_layer(0));
        assert!(!tc.is_sliding_layer(3));
        assert_eq!(tc.head_dim_for_layer(0), 64);
        // Full-attention layer uses global_head_dim when present.
        assert_eq!(tc.head_dim_for_layer(3), 128);
        assert_eq!(tc.rope_params_for_layer(0).rope_type, "default");
        assert_eq!(tc.rope_params_for_layer(3).rope_type, "proportional");
    }
}
