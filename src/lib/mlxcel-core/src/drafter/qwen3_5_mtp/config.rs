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

//! Configuration for the Qwen 3.5 MTP drafter (`model_type: "qwen3_5_mtp"`).
//!
//! Mirrors the upstream `Qwen3_5MTPConfig` shape: a small top level
//! (`block_size`, `tie_word_embeddings`) plus a nested `text_config` that is a
//! full copy of the target's text config. See
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_5_mtp/config.py.
//!
//! The drafter exercises only a subset of the nested text config (the MTP
//! head is a single full-attention decoder layer sized on the target's
//! `hidden_size`), so [`Qwen35MtpTextConfig`] deserializes just that subset
//! and ignores the rest. The full target-side `Qwen35Config` lives in the
//! `mlxcel` binary crate, above `mlxcel-core` in the dependency graph, so it
//! cannot be reused here; field names, defaults, and the
//! `rope_parameters`-derived accessors mirror it exactly.

use serde::Deserialize;

/// Quantization arguments for a (hypothetical) quantized drafter checkpoint.
/// The published `mlx-community/Qwen3.8-27B-MTP-bf16` drafter is unquantized;
/// the field exists so a future quantized split loads without a config edit.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MtpQuantization {
    pub group_size: i32,
    pub bits: i32,
}

/// Subset of the target-mirroring `text_config` the MTP drafter consumes.
///
/// Unknown fields (`layer_types`, `linear_*`, MoE knobs, …) are ignored: the
/// MTP layer is always full attention (upstream builds it with
/// `full_attention_interval=1`), so none of the linear-attention or layout
/// fields participate.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MtpTextConfig {
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    /// RoPE parameters in the same free-form dict shape the target's
    /// `Qwen35Config` parses (`rope_theta`, `partial_rotary_factor`, plus
    /// MRoPE keys the drafter ignores).
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    /// Number of MTP decoder layers. `1` on every published Qwen 3.5 / 3.6 /
    /// 3.8 checkpoint.
    #[serde(default = "default_mtp_num_hidden_layers")]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<Qwen35MtpQuantization>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_mtp_num_hidden_layers() -> usize {
    1
}

impl Qwen35MtpTextConfig {
    /// Effective quantization group size (matches `Qwen35Config::group_size`).
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    /// Effective quantization bit width (matches `Qwen35Config::bits`).
    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// RoPE base frequency. Default mirrors the target-side
    /// `Qwen35Config::rope_theta` fallback.
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(100_000.0)
    }

    /// Partial rotary factor. Default mirrors the target-side
    /// `Qwen35Config::partial_rotary_factor` fallback.
    pub fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.25)
    }

    /// Per-head dimension, falling back to `hidden_size / num_attention_heads`.
    pub fn head_dim_resolved(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Rotary dimensions per head (`head_dim * partial_rotary_factor`),
    /// matching the target-side `Qwen35Config::rope_dims`.
    pub fn rope_dims(&self) -> i32 {
        (self.head_dim_resolved() as f32 * self.partial_rotary_factor()) as i32
    }

    /// Whether the mirrored target family is a MoE variant. The dense MTP
    /// drafter does not implement the MoE decoder layer, so loaders reject
    /// this early with a named error instead of failing on a missing
    /// `switch_mlp` weight.
    pub fn is_moe(&self) -> bool {
        self.model_type.contains("moe")
    }
}

/// Drafter config for the Qwen 3.5 MTP head.
///
/// Mirrors upstream `Qwen3_5MTPConfig`: `block_size` defaults to
/// `mtp_num_hidden_layers + 2` when absent (the **total** verify-round budget
/// including the bonus token, so the published `block_size: 3` checkpoints
/// draft 2 tokens per round).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MtpConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Total round budget including the bonus token. `None` in JSON resolves
    /// to `mtp_num_hidden_layers + 2` in [`Self::normalize`], mirroring
    /// upstream `from_dict`'s `flat.setdefault("block_size", mtp_depth + 2)`.
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub text_config: Option<Qwen35MtpTextConfig>,
}

fn default_model_type() -> String {
    "qwen3_5_mtp".to_string()
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl Qwen35MtpConfig {
    /// Validate and apply the upstream post-init fixups:
    ///
    /// - `text_config` must be present (upstream raises `ValueError`).
    /// - `block_size` defaults to `mtp_num_hidden_layers + 2`.
    /// - `tie_word_embeddings` follows `text_config.tie_word_embeddings`
    ///   (upstream `__post_init__`).
    /// - MoE text configs are rejected: the dense drafter has no
    ///   `Qwen3_5MoeDecoderLayer` port.
    pub fn normalize(mut self) -> Result<Self, String> {
        let text_cfg = self
            .text_config
            .as_ref()
            .ok_or_else(|| "Qwen35MtpConfig.text_config must be set".to_string())?;
        if text_cfg.is_moe() {
            return Err(format!(
                "qwen3_5_mtp drafter: MoE text_config (model_type {:?}) is not supported; \
                 only the dense MTP decoder layer is implemented",
                text_cfg.model_type
            ));
        }
        if self.block_size.is_none() {
            self.block_size = Some(text_cfg.mtp_num_hidden_layers + 2);
        }
        self.tie_word_embeddings = text_cfg.tie_word_embeddings;
        Ok(self)
    }

    /// Nested text config accessor. Call after [`Self::normalize`].
    pub fn text_config(&self) -> &Qwen35MtpTextConfig {
        self.text_config
            .as_ref()
            .expect("Qwen35MtpConfig.text_config must be set (call normalize first)")
    }

    /// Resolved total round budget (bonus token included). Call after
    /// [`Self::normalize`].
    pub fn block_size(&self) -> usize {
        self.block_size
            .expect("Qwen35MtpConfig.block_size resolved by normalize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/Qwen3.8-27B-MTP-bf16` config shape:
    /// `block_size: 3` at top level, `text_config` mirroring the target
    /// (with keys the drafter ignores), `mtp_num_hidden_layers: 1`.
    #[test]
    fn parses_published_qwen38_drafter_config_shape() {
        let json = r#"{
            "block_size": 3,
            "model_type": "qwen3_5_mtp",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "layer_types": ["linear_attention", "full_attention"],
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "num_hidden_layers": 64,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "vocab_size": 248320
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse config");
        let cfg = cfg.normalize().expect("normalize");
        assert_eq!(cfg.model_type, "qwen3_5_mtp");
        assert_eq!(cfg.block_size(), 3);
        assert!(!cfg.tie_word_embeddings);
        let tc = cfg.text_config();
        assert_eq!(tc.hidden_size, 5120);
        assert_eq!(tc.num_attention_heads, 24);
        assert_eq!(tc.num_key_value_heads, 4);
        assert_eq!(tc.head_dim_resolved(), 256);
        assert_eq!(tc.mtp_num_hidden_layers, 1);
        assert_eq!(tc.vocab_size, 248320);
        assert_eq!(tc.rope_theta(), 10_000_000.0);
        // 256 * 0.25 = 64 rotary dims — the partial-rotary geometry the
        // target's full-attention layers use.
        assert_eq!(tc.rope_dims(), 64);
    }

    /// `block_size` omitted resolves to `mtp_num_hidden_layers + 2`
    /// (upstream `from_dict` setdefault).
    #[test]
    fn block_size_defaults_to_mtp_depth_plus_two() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "mtp_num_hidden_layers": 1
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let cfg = cfg.normalize().expect("normalize");
        assert_eq!(cfg.block_size(), 3);
    }

    #[test]
    fn normalize_rejects_missing_text_config() {
        let cfg: Qwen35MtpConfig =
            serde_json::from_str(r#"{"model_type": "qwen3_5_mtp"}"#).expect("parse");
        let err = cfg.normalize().expect_err("must reject");
        assert!(err.contains("text_config"));
    }

    #[test]
    fn normalize_rejects_moe_text_config() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let err = cfg.normalize().expect_err("must reject MoE");
        assert!(err.contains("MoE"), "got: {err}");
    }
}
