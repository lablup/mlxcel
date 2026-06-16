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

//! [`DFlashConfig`] — Rust port of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/config.py.
//!
//! Defaults mirror the published `z-lab/Qwen3.5-4B-DFlash` checkpoint:
//!
//! - `hidden_size = 2560` (drafter hidden dim; smaller than the 4B target's 2560).
//! - `intermediate_size = 9728`.
//! - `num_hidden_layers = 5`.
//! - `num_attention_heads = 32`, `num_key_value_heads = 8`, `head_dim = 128`.
//! - `rms_norm_eps = 1e-6`.
//! - `vocab_size = 248320`.
//! - `rope_theta = 10_000_000.0`.
//! - `tie_word_embeddings = true`.
//! - `block_size = 16`, `mask_token_id = 248070`.
//! - `target_layer_ids = [1, 8, 15, 22, 29]` over `num_target_layers = 32`.
//!
//! The `from_hf_dict` constructor also accepts a nested `dflash_config`
//! sub-object (matching the upstream Python loader) so checkpoints that
//! nest DFlash-specific fields under a `dflash_config` key still load.
//!
//! Upstream reference: 38 lines of Python including the JSON-flattening
//! convention. The Rust port keeps the same flattening semantics for
//! `mask_token_id` and `target_layer_ids`.

use serde::{Deserialize, Serialize};

/// Upstream DFlash default target-layer capture list.
///
/// This is the default in
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/config.py
/// for the original `z-lab/Qwen3.5-4B-DFlash` drafter. Runtime code should
/// prefer the loaded checkpoint's [`DFlashConfig::target_layer_ids`]; this
/// constant exists only as the config default / backwards-compatible fallback.
pub const DEFAULT_TARGET_LAYER_IDS: &[usize] = &[1, 8, 15, 22, 29];

/// Configuration for [`super::model::DFlashDraftModel`].
///
/// Fields are defaulted to match the upstream `DFlashConfig` Python
/// dataclass exactly. The struct is `#[non_exhaustive]` so new upstream
/// fields can be added without breaking external `match` patterns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DFlashConfig {
    /// Drafter hidden size (also the per-target-layer projection dim).
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// MLP intermediate size (SwiGLU gate / up / down inner dim).
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of transformer layers in the drafter.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Number of K/V heads (GQA).
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    /// Per-head dimension; total Q dim is `num_attention_heads * head_dim`.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// RMS-norm epsilon for both attention norms and the pre/post layer
    /// norms (matches upstream).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// Tokenizer vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Max positional length (used as an upper bound on RoPE wrap).
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Whether Q/K/V/O projections carry biases (DFlash uses `false`).
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,

    /// Whether the LM head shares weights with `embed_tokens`.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    /// Default proposal block length per round (caller may override).
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// Placeholder token id stamped into the proposal positions of the
    /// masked forward. The drafter's `embed_tokens.weight` row for this id
    /// is the learnt "mask embedding".
    #[serde(default = "default_mask_token_id")]
    pub mask_token_id: i32,

    /// Indices (over the target's transformer stack) whose hidden states
    /// are concatenated and projected into the drafter's `pre_projection`
    /// input. For Qwen 3.5 these default to `[1, 8, 15, 22, 29]`.
    #[serde(default = "default_target_layer_ids")]
    pub target_layer_ids: Vec<usize>,

    /// Total layer count of the target's transformer stack (32 for Qwen 3.5).
    #[serde(default = "default_num_target_layers")]
    pub num_target_layers: usize,
}

fn default_hidden_size() -> usize {
    2560
}
fn default_intermediate_size() -> usize {
    9728
}
fn default_num_hidden_layers() -> usize {
    5
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_vocab_size() -> usize {
    248320
}
fn default_max_position_embeddings() -> usize {
    262144
}
fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_attention_bias() -> bool {
    false
}
fn default_tie_word_embeddings() -> bool {
    true
}
fn default_block_size() -> usize {
    16
}
fn default_mask_token_id() -> i32 {
    248070
}
fn default_target_layer_ids() -> Vec<usize> {
    DEFAULT_TARGET_LAYER_IDS.to_vec()
}
fn default_num_target_layers() -> usize {
    32
}

impl Default for DFlashConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            head_dim: default_head_dim(),
            rms_norm_eps: default_rms_norm_eps(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            attention_bias: default_attention_bias(),
            tie_word_embeddings: default_tie_word_embeddings(),
            block_size: default_block_size(),
            mask_token_id: default_mask_token_id(),
            target_layer_ids: default_target_layer_ids(),
            num_target_layers: default_num_target_layers(),
        }
    }
}

impl DFlashConfig {
    /// Parse a HuggingFace-style flat dict, also flattening a nested
    /// `dflash_config` sub-object (mirrors upstream `from_dict` / `from_hf_dict`).
    ///
    /// Per the Python loader:
    ///
    /// ```python
    /// flat = dict(params)
    /// dflash_cfg = flat.pop("dflash_config", None) or {}
    /// if "mask_token_id" in dflash_cfg:
    ///     flat["mask_token_id"] = dflash_cfg["mask_token_id"]
    /// if "target_layer_ids" in dflash_cfg:
    ///     flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
    /// ```
    ///
    /// Returns an error if either the input JSON is malformed or required
    /// constraints (e.g. `head_dim * num_attention_heads == hidden_size`
    /// has *not* been enforced — both upstream and this port allow them
    /// to diverge for non-standard configurations) cannot be satisfied.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        // Clone the input so we can rewrite top-level keys from the
        // nested `dflash_config` sub-object.
        let mut flat = match value {
            serde_json::Value::Object(map) => map.clone(),
            _ => {
                return Err(format!(
                    "DFlashConfig::from_json expected a JSON object, got: {value}"
                ));
            }
        };

        if let Some(serde_json::Value::Object(dflash_cfg)) = flat.remove("dflash_config") {
            if let Some(v) = dflash_cfg.get("mask_token_id") {
                flat.insert("mask_token_id".to_string(), v.clone());
            }
            if let Some(v) = dflash_cfg.get("target_layer_ids") {
                flat.insert("target_layer_ids".to_string(), v.clone());
            }
        }

        serde_json::from_value(serde_json::Value::Object(flat))
            .map_err(|e| format!("Failed to parse DFlashConfig: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_matches_upstream_z_lab_qwen35_4b_dflash() {
        let c = DFlashConfig::default();
        // All defaults pinned to the upstream Python dataclass.
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(c.intermediate_size, 9728);
        assert_eq!(c.num_hidden_layers, 5);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(c.vocab_size, 248320);
        assert_eq!(c.max_position_embeddings, 262144);
        assert!((c.rope_theta - 10_000_000.0).abs() < 1e-3);
        assert!(!c.attention_bias);
        assert!(c.tie_word_embeddings);
        assert_eq!(c.block_size, 16);
        assert_eq!(c.mask_token_id, 248070);
        assert_eq!(c.target_layer_ids, vec![1, 8, 15, 22, 29]);
        assert_eq!(c.num_target_layers, 32);
    }

    #[test]
    fn from_json_uses_defaults_when_unset() {
        let cfg = DFlashConfig::from_json(&json!({})).unwrap();
        assert_eq!(cfg, DFlashConfig::default());
    }

    #[test]
    fn from_json_flattens_nested_dflash_config_overrides() {
        // Upstream invariant: a nested `dflash_config` block must override
        // the corresponding top-level fields (mask_token_id, target_layer_ids).
        let cfg = DFlashConfig::from_json(&json!({
            "hidden_size": 2560,
            "mask_token_id": 1,
            "dflash_config": {
                "mask_token_id": 99999,
                "target_layer_ids": [0, 5, 10],
            },
        }))
        .unwrap();
        assert_eq!(cfg.mask_token_id, 99999);
        assert_eq!(cfg.target_layer_ids, vec![0, 5, 10]);
    }

    #[test]
    fn from_json_passes_through_top_level_when_no_nested_block() {
        let cfg = DFlashConfig::from_json(&json!({
            "block_size": 24,
            "mask_token_id": 12345,
        }))
        .unwrap();
        assert_eq!(cfg.block_size, 24);
        assert_eq!(cfg.mask_token_id, 12345);
        // Other fields still default.
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 5);
    }

    #[test]
    fn from_json_rejects_non_object_root() {
        let err = DFlashConfig::from_json(&json!([1, 2, 3])).unwrap_err();
        assert!(err.contains("expected a JSON object"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_malformed_field_type() {
        let err = DFlashConfig::from_json(&json!({
            "block_size": "not a number",
        }))
        .unwrap_err();
        assert!(err.contains("Failed to parse DFlashConfig"));
    }

    #[test]
    fn config_roundtrips_through_json() {
        let original = DFlashConfig::default();
        let serialized = serde_json::to_value(&original).unwrap();
        let restored: DFlashConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored, original);
    }

    /// Pin the invariant that the default `mask_token_id` fits within
    /// the default `vocab_size`. Upstream relies on this to embed the
    /// mask placeholder token via `embed_tokens(mask_id)` directly,
    /// without a special-case codepath.
    #[test]
    fn default_mask_token_id_fits_in_vocab_size() {
        let cfg = DFlashConfig::default();
        assert!(
            (cfg.mask_token_id as usize) < cfg.vocab_size,
            "mask_token_id={} must be < vocab_size={} (the embedding lookup \
             would otherwise index out of bounds)",
            cfg.mask_token_id,
            cfg.vocab_size,
        );
    }

    /// Pin the invariant that `len(target_layer_ids) == num_target_layers / N`
    /// is reasonable — concretely, that every target_layer_id is within
    /// `[0, num_target_layers)`. A misconfigured `target_layer_ids` would
    /// cause the target's hidden-capture path to silently drop a slot,
    /// leading to a degenerate `concat()` shape on the drafter side.
    #[test]
    fn default_target_layer_ids_are_in_bounds() {
        let cfg = DFlashConfig::default();
        for &id in &cfg.target_layer_ids {
            assert!(
                id < cfg.num_target_layers,
                "target_layer_id={id} must be < num_target_layers={}; \
                 default config is malformed",
                cfg.num_target_layers,
            );
        }
    }

    /// Pin the (asymmetric) projection-dim relationship for the
    /// published `z-lab/Qwen3.5-4B-DFlash` defaults: q_proj projects
    /// `hidden_size = 2560` → `n_heads * head_dim = 4096`, then
    /// `o_proj` projects `4096` → `hidden_size = 2560`. The Q output
    /// dim and the residual stream dim DO NOT match in upstream — this
    /// is intentional and the `DFlashAttention::forward` reshape/SDPA/
    /// projection sequence depends on it.
    #[test]
    fn default_q_proj_out_dim_matches_n_heads_times_head_dim() {
        let cfg = DFlashConfig::default();
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim,
            4096,
            "default config: q_proj out dim must equal 4096 (32 heads × 128 head_dim)",
        );
        assert_eq!(
            cfg.hidden_size, 2560,
            "default config: residual stream hidden_size must equal 2560",
        );
        // The two dimensions DIFFER: this is the asymmetric upstream config.
        assert_ne!(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            "default DFlashConfig is asymmetric: q out (4096) != hidden_size (2560). \
             A future patch that 'fixes' this is a regression."
        );
    }
}
