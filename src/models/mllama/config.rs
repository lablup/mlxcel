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

//! Configuration for Llama 3.2 Vision (`mllama`).
//!
//! Faithful port of
//! `references/mlx-vlm/mlx_vlm/models/mllama/config.py`. The checkpoint's
//! `config.json` nests a `text_config` (Llama-3 backbone dimensions plus the
//! `cross_attention_layers` list) and a `vision_config` (the tiled ViT tower).

use serde::Deserialize;

/// Text (language) backbone configuration.
///
/// This is a Llama-3 decoder whose layers at `cross_attention_layers` are
/// replaced by gated cross-attention adapters that attend to the vision
/// tower's features.
#[derive(Debug, Clone, Deserialize)]
pub struct MllamaTextConfig {
    #[serde(default = "default_text_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_cross_attention_layers")]
    pub cross_attention_layers: Vec<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub quantization: Option<crate::models::llama3::Quantization>,
}

impl MllamaTextConfig {
    /// Head dimension (`hidden_size / num_attention_heads` unless overridden).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// `true` when `layer_idx` is a gated cross-attention adapter layer.
    pub fn is_cross_attention_layer(&self, layer_idx: usize) -> bool {
        self.cross_attention_layers.contains(&layer_idx)
    }

    /// Build the `llama3::ModelArgs` that describes the self-attention decoder
    /// layers. The self-attention layers are byte-for-byte the standard Llama-3
    /// block (fused QKV, plain RoPE with `base = rope_theta`), so they reuse the
    /// existing `llama3::TransformerBlock` loader.
    pub fn to_llama3_args(&self) -> crate::models::llama3::ModelArgs {
        let mut text_config = serde_json::json!({
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "rms_norm_eps": self.rms_norm_eps,
            "vocab_size": self.vocab_size,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
        });
        if let Some(q) = &self.quantization {
            text_config["quantization"] =
                serde_json::json!({ "group_size": q.group_size, "bits": q.bits });
        }
        serde_json::from_value(text_config)
            .expect("MllamaTextConfig always produces a valid llama3::ModelArgs")
    }
}

/// Vision tower configuration (tiled ViT with aspect-ratio embeddings).
#[derive(Debug, Clone, Deserialize)]
pub struct MllamaVisionConfig {
    #[serde(default = "default_vision_image_size")]
    pub image_size: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_max_num_tiles")]
    pub max_num_tiles: usize,
    #[serde(default = "default_vision_max_aspect_ratio_id")]
    pub max_aspect_ratio_id: usize,
    #[serde(default = "default_vision_num_global_layers")]
    pub num_global_layers: usize,
    #[serde(default = "default_vision_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_vision_output_dim")]
    pub vision_output_dim: usize,
    #[serde(default = "default_intermediate_layers_indices")]
    pub intermediate_layers_indices: Vec<usize>,
    #[serde(default = "default_supported_aspect_ratios")]
    pub supported_aspect_ratios: Vec<Vec<usize>>,
    /// Quantization block inherited from the checkpoint's top-level
    /// `quantization` (the vision tower, projector, and text backbone are
    /// quantized together). `None` for an unquantized tower.
    #[serde(default)]
    pub quantization: Option<crate::models::llama3::Quantization>,
}

impl MllamaVisionConfig {
    /// Patches per tile including the prepended class token
    /// (`(image_size / patch_size)^2 + 1`).
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2) + 1
    }

    /// Head dimension of the vision self-attention.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Quantization group size, or the MLX default (64) when the tower is not
    /// quantized. Consulted only on the quantized path, where the checkpoint's
    /// top-level `quantization` is inherited into this config.
    pub fn quant_group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    /// Quantization bit width, or the MLX default (4) when the tower is not
    /// quantized.
    pub fn quant_bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

/// Top-level `mllama` configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MllamaConfig {
    pub text_config: MllamaTextConfig,
    pub vision_config: MllamaVisionConfig,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: i32,
    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: i32,
}

fn default_model_type() -> String {
    "mllama".to_string()
}
fn default_text_model_type() -> String {
    "mllama".to_string()
}
fn default_vocab_size() -> usize {
    128256
}
fn default_hidden_size() -> usize {
    4096
}
fn default_intermediate_size() -> usize {
    14336
}
fn default_num_hidden_layers() -> usize {
    40
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    500000.0
}
fn default_tie_word_embeddings() -> bool {
    false
}
fn default_cross_attention_layers() -> Vec<usize> {
    vec![3, 8, 13, 18, 23, 28, 33, 38]
}
fn default_image_token_index() -> i32 {
    128256
}
fn default_vision_feature_layer() -> i32 {
    -2
}

fn default_vision_image_size() -> usize {
    560
}
fn default_vision_patch_size() -> usize {
    14
}
fn default_vision_num_channels() -> usize {
    3
}
fn default_vision_hidden_size() -> usize {
    1280
}
fn default_vision_intermediate_size() -> usize {
    5120
}
fn default_vision_num_hidden_layers() -> usize {
    32
}
fn default_vision_num_attention_heads() -> usize {
    16
}
fn default_vision_max_num_tiles() -> usize {
    4
}
fn default_vision_max_aspect_ratio_id() -> usize {
    8
}
fn default_vision_num_global_layers() -> usize {
    8
}
fn default_vision_norm_eps() -> f32 {
    1e-5
}
fn default_vision_output_dim() -> usize {
    7680
}
fn default_intermediate_layers_indices() -> Vec<usize> {
    vec![3, 7, 15, 23, 30]
}
fn default_supported_aspect_ratios() -> Vec<Vec<usize>> {
    vec![
        vec![1, 1],
        vec![1, 2],
        vec![1, 3],
        vec![1, 4],
        vec![2, 1],
        vec![2, 2],
        vec![3, 1],
        vec![4, 1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_mllama_config() {
        let cfg: MllamaConfig = serde_json::from_str(
            r#"{
                "model_type": "mllama",
                "image_token_index": 128256,
                "text_config": {
                    "model_type": "mllama",
                    "hidden_size": 4096,
                    "num_hidden_layers": 40,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38],
                    "rope_theta": 500000.0
                },
                "vision_config": {
                    "image_size": 560,
                    "patch_size": 14,
                    "hidden_size": 1280,
                    "num_hidden_layers": 32,
                    "num_global_layers": 8,
                    "vision_output_dim": 7680,
                    "intermediate_layers_indices": [3, 7, 15, 23, 30]
                }
            }"#,
        )
        .expect("valid mllama config");

        assert_eq!(cfg.text_config.num_hidden_layers, 40);
        assert!(cfg.text_config.is_cross_attention_layer(3));
        assert!(!cfg.text_config.is_cross_attention_layer(0));
        assert_eq!(cfg.text_config.head_dim(), 128);
        assert_eq!(cfg.vision_config.num_patches(), 1601);
        assert_eq!(cfg.vision_config.head_dim(), 80);
        assert_eq!(cfg.vision_config.supported_aspect_ratios.len(), 8);
    }

    #[test]
    fn vision_config_reads_inherited_quantization() {
        // Unquantized tower: MLX defaults, not a quantized load.
        let dense: MllamaVisionConfig = serde_json::from_str("{}").expect("defaults");
        assert!(dense.quantization.is_none());
        assert_eq!(dense.quant_group_size(), 64);
        assert_eq!(dense.quant_bits(), 4);

        // Quantized tower: the top-level block is inherited into vision_config
        // by the loader before parsing.
        let quant: MllamaVisionConfig =
            serde_json::from_str(r#"{ "quantization": { "group_size": 64, "bits": 4 } }"#)
                .expect("quantized vision config");
        assert_eq!(quant.quant_group_size(), 64);
        assert_eq!(quant.quant_bits(), 4);
    }

    #[test]
    fn text_config_defaults_match_reference() {
        let cfg: MllamaTextConfig = serde_json::from_str("{}").expect("defaults");
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(
            cfg.cross_attention_layers,
            vec![3, 8, 13, 18, 23, 28, 33, 38]
        );
        assert!(!cfg.tie_word_embeddings);
    }
}
