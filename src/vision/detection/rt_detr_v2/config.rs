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

//! RT-DETRv2 configuration.
//!
//! Mirrors the HuggingFace `RTDetrV2Config` schema
//! (`transformers.models.rt_detr_v2.configuration_rt_detr_v2`) and the
//! upstream mlx-vlm port in
//! `references/mlx-vlm/mlx_vlm/models/rt_detr_v2/config.py`. The HF
//! `config.json` stores backbone / hybrid-encoder / decoder fields flat at
//! the top level; this module parses them into one [`RtDetrV2Config`] and
//! exposes typed sub-views (`backbone()`, `encoder()`, `transformer()`).

use std::collections::BTreeMap;

use serde::Deserialize;

/// Default for [`RtDetrV2Config::image_size`] when neither `image_size` nor a
/// `size.{height,width}` block is present.
const DEFAULT_IMAGE_SIZE: usize = 640;

fn default_image_size() -> usize {
    DEFAULT_IMAGE_SIZE
}
fn default_num_labels() -> usize {
    17
}
fn default_d_model() -> usize {
    256
}
fn default_encoder_hidden_dim() -> usize {
    256
}
fn default_encoder_in_channels() -> Vec<usize> {
    vec![512, 1024, 2048]
}
fn default_feat_strides() -> Vec<usize> {
    vec![8, 16, 32]
}
fn default_encoder_layers() -> usize {
    1
}
fn default_encoder_ffn_dim() -> usize {
    1024
}
fn default_encoder_attention_heads() -> usize {
    8
}
fn default_encoder_activation() -> String {
    "gelu".to_string()
}
fn default_encode_proj_layers() -> Vec<usize> {
    vec![2]
}
fn default_positional_encoding_temperature() -> f32 {
    10000.0
}
fn default_activation() -> String {
    "silu".to_string()
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_hidden_expansion() -> f32 {
    1.0
}
fn default_batch_norm_eps() -> f32 {
    1e-5
}
fn default_decoder_layers() -> usize {
    6
}
fn default_decoder_attention_heads() -> usize {
    8
}
fn default_decoder_ffn_dim() -> usize {
    1024
}
fn default_decoder_in_channels() -> Vec<usize> {
    vec![256, 256, 256]
}
fn default_decoder_activation() -> String {
    "relu".to_string()
}
fn default_decoder_method() -> String {
    "default".to_string()
}
fn default_decoder_n_levels() -> usize {
    3
}
fn default_decoder_n_points() -> usize {
    4
}
fn default_decoder_offset_scale() -> f32 {
    0.5
}
fn default_num_feature_levels() -> usize {
    3
}
fn default_num_queries() -> usize {
    300
}

/// ResNet-vd backbone configuration.
///
/// The `vd` variant has a 3-stage stem (3x3 -> 3x3 -> 3x3, stride 2/1/1)
/// followed by a 3x3 stride-2 max-pool, and uses `AvgPool2x2 stride 2 + 1x1
/// conv` for downsampling shortcuts. Depths default to ResNet-50
/// (`[3, 4, 6, 3]`); ResNet-101 is `[3, 4, 23, 3]`.
#[derive(Debug, Clone, Deserialize)]
pub struct BackboneConfig {
    #[serde(default = "BackboneConfig::default_depths")]
    pub depths: Vec<usize>,
    #[serde(default)]
    pub downsample_in_bottleneck: bool,
    #[serde(default)]
    pub downsample_in_first_stage: bool,
    #[serde(default = "BackboneConfig::default_embedding_size")]
    pub embedding_size: usize,
    #[serde(default = "BackboneConfig::default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "BackboneConfig::default_hidden_sizes")]
    pub hidden_sizes: Vec<usize>,
    #[serde(default = "BackboneConfig::default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "BackboneConfig::default_out_features")]
    pub out_features: Vec<String>,
}

impl BackboneConfig {
    fn default_depths() -> Vec<usize> {
        vec![3, 4, 6, 3]
    }
    fn default_embedding_size() -> usize {
        64
    }
    fn default_hidden_act() -> String {
        "relu".to_string()
    }
    fn default_hidden_sizes() -> Vec<usize> {
        vec![256, 512, 1024, 2048]
    }
    fn default_num_channels() -> usize {
        3
    }
    fn default_out_features() -> Vec<String> {
        vec![
            "stage2".to_string(),
            "stage3".to_string(),
            "stage4".to_string(),
        ]
    }

    /// Zero-based stage indices selected by `out_features` (e.g. `stage2` ->
    /// index 1). Stage 0 (stride 4) is computed but typically dropped.
    pub fn out_stage_indices(&self) -> Vec<usize> {
        self.out_features
            .iter()
            .filter_map(|name| name.strip_prefix("stage"))
            .filter_map(|n| n.parse::<usize>().ok())
            .map(|n| n - 1)
            .collect()
    }
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            depths: Self::default_depths(),
            downsample_in_bottleneck: false,
            downsample_in_first_stage: false,
            embedding_size: Self::default_embedding_size(),
            hidden_act: Self::default_hidden_act(),
            hidden_sizes: Self::default_hidden_sizes(),
            num_channels: Self::default_num_channels(),
            out_features: Self::default_out_features(),
        }
    }
}

/// `size: {"height": H, "width": W}` block from `preprocessor_config.json` or
/// `config.json`. We require square inputs, so only the height is used.
#[derive(Debug, Clone, Deserialize)]
struct SizeBlock {
    #[serde(default)]
    height: Option<usize>,
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    shortest_edge: Option<usize>,
}

/// Top-level RT-DETRv2 configuration parsed from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RtDetrV2Config {
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    /// HF stores the resize target under `size: {height, width}` in some
    /// checkpoints; captured here so [`Self::resolve`] can fold it into
    /// `image_size`.
    #[serde(default)]
    size: Option<SizeBlock>,

    #[serde(default = "default_num_labels")]
    pub num_labels: usize,
    #[serde(default)]
    pub id2label: Option<BTreeMap<String, String>>,

    #[serde(default)]
    pub backbone_config: BackboneConfig,

    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_encoder_hidden_dim")]
    pub encoder_hidden_dim: usize,
    #[serde(default = "default_encoder_in_channels")]
    pub encoder_in_channels: Vec<usize>,
    #[serde(default = "default_feat_strides")]
    pub feat_strides: Vec<usize>,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_encoder_activation")]
    pub encoder_activation_function: String,
    #[serde(default = "default_encode_proj_layers")]
    pub encode_proj_layers: Vec<usize>,
    #[serde(default = "default_positional_encoding_temperature")]
    pub positional_encoding_temperature: f32,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default)]
    pub normalize_before: bool,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_hidden_expansion")]
    pub hidden_expansion: f32,
    #[serde(default = "default_batch_norm_eps")]
    pub batch_norm_eps: f32,

    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: usize,
    #[serde(default = "default_decoder_attention_heads")]
    pub decoder_attention_heads: usize,
    #[serde(default = "default_decoder_ffn_dim")]
    pub decoder_ffn_dim: usize,
    #[serde(default = "default_decoder_in_channels")]
    pub decoder_in_channels: Vec<usize>,
    #[serde(default = "default_decoder_activation")]
    pub decoder_activation_function: String,
    #[serde(default = "default_decoder_method")]
    pub decoder_method: String,
    #[serde(default = "default_decoder_n_levels")]
    pub decoder_n_levels: usize,
    #[serde(default = "default_decoder_n_points")]
    pub decoder_n_points: usize,
    #[serde(default = "default_decoder_offset_scale")]
    pub decoder_offset_scale: f32,
    #[serde(default = "default_num_feature_levels")]
    pub num_feature_levels: usize,
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,
}

impl RtDetrV2Config {
    /// Parse a `config.json` string into a resolved config.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let mut cfg: RtDetrV2Config =
            serde_json::from_str(s).map_err(|e| format!("RT-DETRv2 config parse error: {e}"))?;
        cfg.resolve();
        Ok(cfg)
    }

    /// Fold the optional `size` block into `image_size`. Called once after
    /// deserialization.
    fn resolve(&mut self) {
        if let Some(size) = &self.size
            && let Some(h) = size.height.or(size.shortest_edge).or(size.width)
        {
            self.image_size = h;
        }
    }

    /// Validate cross-field invariants the forward path relies on. Mirrors the
    /// `raise` checks scattered through the Python modules so a malformed
    /// checkpoint fails at load time with a clear message rather than mid-graph.
    pub fn validate(&self) -> Result<(), String> {
        if !self.d_model.is_multiple_of(self.decoder_attention_heads) {
            return Err(format!(
                "d_model ({}) must be divisible by decoder_attention_heads ({})",
                self.d_model, self.decoder_attention_heads
            ));
        }
        if !self
            .encoder_hidden_dim
            .is_multiple_of(self.encoder_attention_heads)
        {
            return Err(format!(
                "encoder_hidden_dim ({}) must be divisible by encoder_attention_heads ({})",
                self.encoder_hidden_dim, self.encoder_attention_heads
            ));
        }
        if !self.encoder_hidden_dim.is_multiple_of(4) {
            return Err(format!(
                "encoder_hidden_dim ({}) must be divisible by 4 for the sine \
                 position embedding",
                self.encoder_hidden_dim
            ));
        }
        if self.decoder_method != "default" && self.decoder_method != "discrete" {
            return Err(format!(
                "Unsupported decoder_method {:?}; expected 'default' or 'discrete'",
                self.decoder_method
            ));
        }
        if self.encoder_in_channels.len() != self.backbone_config.out_features.len() {
            return Err(format!(
                "encoder_in_channels ({}) must match backbone out_features ({})",
                self.encoder_in_channels.len(),
                self.backbone_config.out_features.len()
            ));
        }
        Ok(())
    }

    /// Backbone sub-config.
    pub fn backbone(&self) -> &BackboneConfig {
        &self.backbone_config
    }

    /// Number of feature-pyramid levels (one per backbone output feature).
    pub fn num_levels(&self) -> usize {
        self.encoder_in_channels.len()
    }

    /// Ordered class-name list resolved from `id2label`, sorted by integer id.
    /// `None` when the checkpoint provides no label map (numeric fallback at
    /// decode time).
    pub fn class_names(&self) -> Option<Vec<String>> {
        let map = self.id2label.as_ref()?;
        if map.is_empty() {
            return None;
        }
        let mut entries: Vec<(i64, String)> = map
            .iter()
            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v.clone())))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        Some(entries.into_iter().map(|(_, v)| v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCLING_CONFIG: &str = r#"{
        "model_type": "rt_detr_v2",
        "architectures": ["RTDetrV2ForObjectDetection"],
        "backbone_config": {
            "depths": [3, 4, 6, 3],
            "embedding_size": 64,
            "hidden_sizes": [256, 512, 1024, 2048],
            "out_features": ["stage2", "stage3", "stage4"]
        },
        "d_model": 256,
        "encoder_hidden_dim": 256,
        "encoder_in_channels": [512, 1024, 2048],
        "num_labels": 17,
        "num_queries": 300,
        "decoder_layers": 6,
        "id2label": {"0": "caption", "1": "footnote", "10": "title", "2": "formula"}
    }"#;

    #[test]
    fn parses_docling_config() {
        let cfg = RtDetrV2Config::from_json_str(DOCLING_CONFIG).unwrap();
        assert_eq!(cfg.d_model, 256);
        assert_eq!(cfg.num_queries, 300);
        assert_eq!(cfg.num_labels, 17);
        assert_eq!(cfg.backbone_config.depths, vec![3, 4, 6, 3]);
        assert_eq!(cfg.num_levels(), 3);
        cfg.validate().unwrap();
    }

    #[test]
    fn class_names_sorted_by_int_id() {
        let cfg = RtDetrV2Config::from_json_str(DOCLING_CONFIG).unwrap();
        let names = cfg.class_names().unwrap();
        // Keys 0,1,2,10 -> sorted numerically, not lexically.
        assert_eq!(names, vec!["caption", "footnote", "formula", "title"]);
    }

    #[test]
    fn out_stage_indices_are_zero_based() {
        let cfg = RtDetrV2Config::from_json_str(DOCLING_CONFIG).unwrap();
        assert_eq!(cfg.backbone_config.out_stage_indices(), vec![1, 2, 3]);
    }

    #[test]
    fn resnet101_depths_via_default_when_absent() {
        let json = r#"{"backbone_config": {"depths": [3, 4, 23, 3]}}"#;
        let cfg = RtDetrV2Config::from_json_str(json).unwrap();
        assert_eq!(cfg.backbone_config.depths, vec![3, 4, 23, 3]);
        // Top-level fields fall back to defaults.
        assert_eq!(cfg.d_model, 256);
        assert_eq!(cfg.image_size, 640);
    }

    #[test]
    fn size_block_folds_into_image_size() {
        let json = r#"{"size": {"height": 800, "width": 800}}"#;
        let cfg = RtDetrV2Config::from_json_str(json).unwrap();
        assert_eq!(cfg.image_size, 800);
    }

    #[test]
    fn validate_rejects_indivisible_d_model() {
        let json = r#"{"d_model": 250, "decoder_attention_heads": 8}"#;
        let cfg = RtDetrV2Config::from_json_str(json).unwrap();
        assert!(cfg.validate().is_err());
    }
}
