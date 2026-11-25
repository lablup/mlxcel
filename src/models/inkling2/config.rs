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

//! Inkling 2 configuration (`inkling2`, `inkling2_loop`).
//!
//! Derived from mlx-vlm `mlx_vlm/models/inkling2/config.py` (MIT).

use serde::Deserialize;

const DEFAULT_EOS_TOKEN_ID: i32 = 200_006;

fn d_text_model_type() -> String {
    "inkling2_text".into()
}
fn d_hidden() -> usize {
    4608
}
fn d_layers() -> usize {
    48
}
fn d_vocab() -> usize {
    201_024
}
fn d_unpadded_vocab() -> Option<usize> {
    Some(200_058)
}
fn d_eps() -> f32 {
    1e-6
}
fn d_heads() -> usize {
    48
}
fn d_kv_heads() -> usize {
    8
}
fn d_head_dim() -> usize {
    128
}
fn d_window() -> usize {
    1024
}
fn d_rel() -> usize {
    16
}
fn d_extent() -> usize {
    1024
}
fn d_log_floor() -> Option<usize> {
    Some(128_000)
}
fn d_log_alpha() -> f32 {
    0.1
}
fn d_kernel() -> usize {
    4
}
fn d_intermediate() -> usize {
    2304
}
fn d_routed() -> usize {
    384
}
fn d_topk() -> usize {
    8
}
fn d_route_scale() -> f32 {
    8.0
}
fn d_qk_clip() -> f32 {
    22.627_417
}
fn d_rel_clip() -> f32 {
    8.0
}
fn d_block() -> usize {
    6
}
fn d_true() -> bool {
    true
}
fn d_gate_softcap() -> f32 {
    4.0
}
fn d_up_softcap() -> f32 {
    25.0
}
fn d_final_softcap() -> f32 {
    30.0
}
fn d_image_token() -> i32 {
    200_054
}

/// Mini's looped core: layers `core_start..core_end` run `num_passes` times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Inkling2LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2TextConfig {
    #[serde(default = "d_text_model_type")]
    pub model_type: String,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_unpadded_vocab")]
    pub unpadded_vocab_size: Option<usize>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_heads")]
    pub swa_num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub swa_num_key_value_heads: usize,
    #[serde(default = "d_head_dim")]
    pub swa_head_dim: usize,
    #[serde(default = "d_window")]
    pub sliding_window_size: usize,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "d_rel")]
    pub d_rel: usize,
    #[serde(default = "d_extent")]
    pub rel_extent: usize,
    #[serde(default = "d_log_floor")]
    pub log_scaling_n_floor: Option<usize>,
    #[serde(default = "d_log_alpha")]
    pub log_scaling_alpha: f32,
    #[serde(default = "d_kernel")]
    pub sconv_kernel_size: usize,
    #[serde(default = "d_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "d_routed")]
    pub n_routed_experts: usize,
    #[serde(default = "d_topk")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    #[serde(default = "d_route_scale")]
    pub route_scale: f32,
    #[serde(default = "d_qk_clip")]
    pub qk_l2_clip_threshold: f32,
    #[serde(default = "d_rel_clip")]
    pub rel_l2_clip_threshold: f32,
    #[serde(default = "d_block")]
    pub n_layers_per_attn_res_block: usize,
    #[serde(default = "d_true")]
    pub attn_res_normalize_values: bool,
    #[serde(default = "d_true")]
    pub use_value_norm_no_gamma: bool,
    #[serde(default = "d_gate_softcap")]
    pub mlp_gate_softcap: f32,
    #[serde(default = "d_up_softcap")]
    pub mlp_up_softcap: f32,
    #[serde(default = "d_final_softcap")]
    pub final_logit_softcapping: f32,
    #[serde(default)]
    pub loop_config: Option<Inkling2LoopConfig>,
}

impl Inkling2TextConfig {
    /// Physical layer index of every visit, in execution order.
    ///
    /// Mini: `0..6, (6..18) x 3, 18..24`, 48 visits over 24 layers.
    pub fn layer_indices(&self) -> Vec<usize> {
        let n = self.num_hidden_layers;
        match self.loop_config {
            None => (0..n).collect(),
            Some(loop_config) => {
                let mut indices: Vec<usize> = (0..loop_config.core_start).collect();
                for _ in 0..loop_config.num_passes {
                    indices.extend(loop_config.core_start..loop_config.core_end);
                }
                indices.extend(loop_config.core_end..n);
                indices
            }
        }
    }

    pub fn layer_is_sliding(&self, index: usize) -> bool {
        if let Some(types) = &self.layer_types {
            return types
                .get(index)
                .is_some_and(|kind| kind == "hybrid_sliding");
        }
        if let Some(ids) = &self.local_layer_ids {
            return ids.contains(&index);
        }
        !(index + 1).is_multiple_of(6)
    }

    /// `(heads, kv_heads, head_dim)` of layer `index`.
    pub fn attention_heads(&self, index: usize) -> (usize, usize, usize) {
        if self.layer_is_sliding(index) {
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
        } else {
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )
        }
    }

    pub fn in_loop_core(&self, index: usize) -> bool {
        self.loop_config
            .is_some_and(|lc| lc.core_start <= index && index < lc.core_end)
    }

    pub fn logits_width(&self) -> usize {
        self.unpadded_vocab_size
            .unwrap_or(self.vocab_size)
            .min(self.vocab_size)
    }

    /// The checks upstream `TextConfig.__post_init__` makes, plus the shape
    /// invariants the ops below rely on.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("vocab_size", self.vocab_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("swa_num_attention_heads", self.swa_num_attention_heads),
            ("swa_num_key_value_heads", self.swa_num_key_value_heads),
            ("swa_head_dim", self.swa_head_dim),
            ("sliding_window_size", self.sliding_window_size),
            ("d_rel", self.d_rel),
            ("rel_extent", self.rel_extent),
            ("sconv_kernel_size", self.sconv_kernel_size),
            ("intermediate_size", self.intermediate_size),
            ("n_routed_experts", self.n_routed_experts),
            (
                "n_layers_per_attn_res_block",
                self.n_layers_per_attn_res_block,
            ),
        ] {
            if value == 0 {
                return Err(format!("Inkling 2 {name} must be positive"));
            }
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
            || !self
                .swa_num_attention_heads
                .is_multiple_of(self.swa_num_key_value_heads)
        {
            return Err("Inkling 2 query-head counts must be divisible by KV-head counts".into());
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            return Err("Inkling 2 num_experts_per_tok must be in 1..=n_routed_experts".into());
        }
        if self.n_shared_experts != 0 {
            return Err(format!(
                "Inkling 2 n_shared_experts must be 0 (no release has shared experts), got {}",
                self.n_shared_experts
            ));
        }
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers {
                return Err("Inkling 2 layer_types length must equal num_hidden_layers".into());
            }
            if let Some(kind) = types
                .iter()
                .find(|kind| !matches!(kind.as_str(), "hybrid_sliding" | "hybrid"))
            {
                return Err(format!(
                    "Inkling 2 layer_types entries must be 'hybrid_sliding' or 'hybrid', got {kind:?}"
                ));
            }
        }
        for (name, value) in [
            ("qk_l2_clip_threshold", self.qk_l2_clip_threshold),
            ("rel_l2_clip_threshold", self.rel_l2_clip_threshold),
            ("route_scale", self.route_scale),
            ("mlp_gate_softcap", self.mlp_gate_softcap),
            ("mlp_up_softcap", self.mlp_up_softcap),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "Inkling 2 {name} must be finite and positive, got {value}"
                ));
            }
        }
        if !self.final_logit_softcapping.is_finite() || self.final_logit_softcapping < 0.0 {
            return Err("Inkling 2 final_logit_softcapping must be finite and non-negative".into());
        }
        if self.log_scaling_n_floor == Some(0) {
            return Err("Inkling 2 log_scaling_n_floor must be positive when set".into());
        }
        // Every bounded axis below feeds an i32 MLX shape.
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("vocab_size", self.vocab_size),
            ("sliding_window_size", self.sliding_window_size),
            ("rel_extent", self.rel_extent),
        ] {
            if value > i32::MAX as usize / 4 {
                return Err(format!(
                    "Inkling 2 {name} {value} exceeds the MLX i32 range"
                ));
            }
        }
        self.validate_loop()
    }

    fn validate_loop(&self) -> Result<(), String> {
        let Some(lc) = self.loop_config else {
            if self.model_type == "inkling2_loop_text" {
                return Err("Inkling 2 Mini (inkling2_loop_text) requires loop_config".into());
            }
            return Ok(());
        };
        let block = self.n_layers_per_attn_res_block;
        let field_error = |field: &str, detail: String| {
            Err(format!(
                "Unsupported Inkling 2 loop_config.{field}: {detail}"
            ))
        };
        if lc.is_varloop {
            return field_error("is_varloop", "variable loops are not supported".into());
        }
        if !(0 < lc.core_start && lc.core_start < lc.core_end) {
            return field_error(
                "core_start",
                format!(
                    "{} must satisfy 0 < core_start < core_end ({})",
                    lc.core_start, lc.core_end
                ),
            );
        }
        if lc.core_end > self.num_hidden_layers {
            return field_error(
                "core_end",
                format!(
                    "{} exceeds num_hidden_layers {}",
                    lc.core_end, self.num_hidden_layers
                ),
            );
        }
        if !lc.core_start.is_multiple_of(block) {
            return field_error(
                "core_start",
                format!(
                    "{} is not a multiple of n_layers_per_attn_res_block {block}",
                    lc.core_start
                ),
            );
        }
        if !lc.core_end.is_multiple_of(block) {
            return field_error(
                "core_end",
                format!(
                    "{} is not a multiple of n_layers_per_attn_res_block {block}",
                    lc.core_end
                ),
            );
        }
        if lc.num_passes == 0 || lc.num_passes > 64 {
            return field_error("num_passes", format!("{} must be in 1..=64", lc.num_passes));
        }
        if lc.gate_rank == 0 {
            return field_error("gate_rank", "must be at least 1".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2VisionConfig {
    #[serde(default = "d_native_patch")]
    pub native_patch_size: usize,
    #[serde(default = "d_merge")]
    pub adapter_merge_size: usize,
    #[serde(default = "d_proj_hidden")]
    pub projection_hidden_dim: usize,
    #[serde(default)]
    pub projector_bias: bool,
    #[serde(default = "d_vision_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_vision_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_vision_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_channels")]
    pub num_channels: usize,
    #[serde(default = "d_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "d_posemb_grid")]
    pub posemb_grid_size: usize,
    #[serde(default = "d_ln_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f32,
}

fn d_native_patch() -> usize {
    20
}
fn d_merge() -> usize {
    2
}
fn d_proj_hidden() -> usize {
    4096
}
fn d_vision_hidden() -> usize {
    1024
}
fn d_vision_layers() -> usize {
    23
}
fn d_vision_heads() -> usize {
    16
}
fn d_channels() -> usize {
    3
}
fn d_mlp_ratio() -> f32 {
    4.0
}
fn d_posemb_grid() -> usize {
    32
}
fn d_ln_eps() -> f32 {
    1e-5
}
fn d_rope_theta() -> f32 {
    10_000.0
}

impl Inkling2VisionConfig {
    pub fn tile_size(&self) -> usize {
        self.native_patch_size * self.adapter_merge_size
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    pub fn mlp_width(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("native_patch_size", self.native_patch_size),
            ("adapter_merge_size", self.adapter_merge_size),
            ("projection_hidden_dim", self.projection_hidden_dim),
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_channels", self.num_channels),
            ("posemb_grid_size", self.posemb_grid_size),
        ] {
            if value == 0 || value > 1 << 20 {
                return Err(format!(
                    "Inkling 2 vision {name} must be in 1..=2^20, got {value}"
                ));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err("Inkling 2 vision hidden_size must divide by num_attention_heads".into());
        }
        // 2-D RoPE splits each head into an x half and a y half of rotated pairs.
        if !self.head_dim().is_multiple_of(4) {
            return Err(format!(
                "Inkling 2 vision head_dim {} must be a multiple of 4 for 2-D RoPE",
                self.head_dim()
            ));
        }
        if !self.mlp_ratio.is_finite() || self.mlp_ratio <= 0.0 || self.mlp_width() == 0 {
            return Err("Inkling 2 vision mlp_ratio must be finite and positive".into());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 1.0 {
            return Err("Inkling 2 vision rope_theta must be finite and above 1".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2Config {
    #[serde(default = "d_model_type")]
    pub model_type: String,
    pub text_config: Inkling2TextConfig,
    #[serde(default)]
    pub vision_config: Option<Inkling2VisionConfig>,
    #[serde(default = "d_image_token")]
    pub image_token_id: i32,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

fn d_model_type() -> String {
    "inkling2".into()
}

impl Inkling2Config {
    pub fn from_json(raw: &str) -> Result<Self, String> {
        let raw = crate::models::sanitize_config_json(raw);
        let config: Self = serde_json::from_str(&raw)
            .map_err(|error| format!("Failed to parse Inkling 2 config: {error}"))?;
        config.text_config.validate()?;
        if let Some(vision) = &config.vision_config {
            vision.validate()?;
        }
        Ok(config)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        let ids = crate::models::parse_optional_eos_token_ids(&self.eos_token_id);
        if ids.is_empty() {
            vec![DEFAULT_EOS_TOKEN_ID]
        } else {
            ids
        }
    }

    /// The mlx-vlm conversion block (`quantization`). The native release's
    /// `quantization_config` (`modelopt` / `NVFP4`) is not one: its per-layer
    /// table lives in `hf_quant_config.json` (see `quant.rs`).
    pub fn mlx_quantization(&self) -> Option<&serde_json::Value> {
        self.quantization.as_ref().filter(|value| !value.is_null())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(extra: serde_json::Value) -> Inkling2TextConfig {
        let mut base = serde_json::json!({
            "model_type": "inkling2_loop_text",
            "hidden_size": 32, "num_hidden_layers": 24,
            "n_layers_per_attn_res_block": 6,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 4}
        });
        for (key, value) in extra.as_object().unwrap() {
            base[key] = value.clone();
        }
        serde_json::from_value(base).unwrap()
    }

    #[test]
    fn mini_layer_indices_visit_the_core_three_times() {
        let config = text(serde_json::json!({}));
        config.validate().unwrap();
        let mut expected: Vec<usize> = (0..6).collect();
        for _ in 0..3 {
            expected.extend(6..18);
        }
        expected.extend(18..24);
        assert_eq!(config.layer_indices(), expected);
        assert_eq!(config.layer_indices().len(), 48);
    }

    #[test]
    fn small_layer_indices_are_the_identity() {
        let config: Inkling2TextConfig = serde_json::from_value(serde_json::json!({
            "model_type": "inkling2_text", "num_hidden_layers": 48
        }))
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.layer_indices(), (0..48).collect::<Vec<_>>());
        assert!(config.layer_is_sliding(0) && !config.layer_is_sliding(5));
    }

    #[test]
    fn loop_config_errors_name_the_field() {
        let unaligned = text(serde_json::json!({
            "loop_config": {"core_start": 4, "core_end": 18, "num_passes": 3, "gate_rank": 4}
        }));
        let error = unaligned.validate().unwrap_err();
        assert!(error.contains("loop_config.core_start"), "{error}");

        let varloop = text(serde_json::json!({
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 4, "is_varloop": true}
        }));
        let error = varloop.validate().unwrap_err();
        assert!(error.contains("loop_config.is_varloop"), "{error}");

        let mut missing = text(serde_json::json!({}));
        missing.loop_config = None;
        assert!(
            missing
                .validate()
                .unwrap_err()
                .contains("requires loop_config")
        );
    }

    #[test]
    fn layer_types_take_precedence_over_local_ids() {
        let config = text(serde_json::json!({
            "num_hidden_layers": 2,
            "loop_config": null,
            "model_type": "inkling2_text",
            "layer_types": ["hybrid", "hybrid_sliding"],
            "local_layer_ids": [0]
        }));
        assert!(!config.layer_is_sliding(0));
        assert!(config.layer_is_sliding(1));
    }
}
