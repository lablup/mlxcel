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

//! Configuration for the released RF-DETR Base and Seg Small checkpoints.

use std::collections::BTreeMap;

use serde::Deserialize;

const BACKBONE_LAYERS: usize = 12;
const MAX_RESOLUTION: usize = 2048;
const MAX_QUERIES: usize = 1024;
const MAX_CLASSES: usize = 1024;
const MAX_DECODER_LAYERS: usize = 12;
const MAX_WINDOWS_PER_AXIS: usize = 16;

/// Flat configuration used by the public mlx-community RF-DETR checkpoints.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RfDetrConfig {
    pub model_type: String,
    pub encoder: String,
    pub hidden_dim: usize,
    pub resolution: usize,
    pub dec_layers: usize,
    pub num_queries: usize,
    pub num_classes: usize,
    pub patch_size: usize,
    pub num_windows: usize,
    pub group_detr: usize,
    pub sa_nheads: usize,
    pub ca_nheads: usize,
    pub dec_n_points: usize,
    pub two_stage: bool,
    pub bbox_reparam: bool,
    pub lite_refpoint_refine: bool,
    pub layer_norm: bool,
    pub out_feature_indexes: Vec<usize>,
    pub projector_scale: Vec<String>,
    pub segmentation: bool,
    pub positional_encoding_size: Option<usize>,
    pub seg_num_blocks: usize,
    pub id2label: Option<BTreeMap<String, String>>,
}

impl Default for RfDetrConfig {
    fn default() -> Self {
        Self {
            model_type: "rf-detr".into(),
            encoder: "dinov2_windowed_small".into(),
            hidden_dim: 256,
            resolution: 560,
            dec_layers: 3,
            num_queries: 300,
            num_classes: 90,
            patch_size: 14,
            num_windows: 4,
            group_detr: 13,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            two_stage: true,
            bbox_reparam: true,
            lite_refpoint_refine: true,
            layer_norm: true,
            out_feature_indexes: vec![2, 5, 8, 11],
            projector_scale: vec!["P4".into()],
            segmentation: false,
            positional_encoding_size: None,
            seg_num_blocks: 4,
            id2label: None,
        }
    }
}

impl RfDetrConfig {
    /// Parse and validate a checkpoint `config.json` document.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(json)
            .map_err(|error| format!("RF-DETR config parse error: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject unsupported variants and resource-amplifying geometry early.
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.model_type.as_str(), "rf-detr" | "rfdetr") {
            return Err(format!(
                "unsupported RF-DETR model_type {:?}",
                self.model_type
            ));
        }
        if self.encoder != "dinov2_windowed_small" {
            return Err(format!("unsupported RF-DETR encoder {:?}", self.encoder));
        }
        if self.resolution == 0 || self.resolution > MAX_RESOLUTION {
            return Err(format!(
                "RF-DETR resolution must be in 1..={MAX_RESOLUTION}; got {}",
                self.resolution
            ));
        }
        if self.num_queries == 0 || self.num_queries > MAX_QUERIES {
            return Err(format!(
                "RF-DETR num_queries must be in 1..={MAX_QUERIES}; got {}",
                self.num_queries
            ));
        }
        if self.num_classes == 0 || self.num_classes > MAX_CLASSES {
            return Err(format!(
                "RF-DETR num_classes must be in 1..={MAX_CLASSES}; got {}",
                self.num_classes
            ));
        }
        if self.dec_layers == 0 || self.dec_layers > MAX_DECODER_LAYERS {
            return Err(format!(
                "RF-DETR dec_layers must be in 1..={MAX_DECODER_LAYERS}; got {}",
                self.dec_layers
            ));
        }
        if self.patch_size == 0
            || self.num_windows == 0
            || self.num_windows > MAX_WINDOWS_PER_AXIS
            || !self.resolution.is_multiple_of(self.patch_size)
            || !(self.resolution / self.patch_size).is_multiple_of(self.num_windows)
        {
            return Err("RF-DETR patch grid must divide evenly into the configured windows".into());
        }
        if self.hidden_dim != 256
            || self.num_classes != 90
            || self.group_detr != 13
            || self.sa_nheads != 8
            || self.ca_nheads != 16
            || self.dec_n_points != 2
            || !self.two_stage
            || !self.bbox_reparam
            || !self.lite_refpoint_refine
            || !self.layer_norm
            || self.projector_scale.len() != 1
            || self.projector_scale[0] != "P4"
            || self.seg_num_blocks != 4
        {
            return Err("unsupported RF-DETR transformer or projector configuration".into());
        }
        if self.out_feature_indexes.len() != 4
            || self
                .out_feature_indexes
                .iter()
                .any(|&index| index == 0 || index > BACKBONE_LAYERS)
            || self
                .out_feature_indexes
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "RF-DETR out_feature_indexes must contain four increasing 1-based layers".into(),
            );
        }

        let supported = if self.segmentation {
            self.resolution == 384
                && self.patch_size == 12
                && self.num_windows == 2
                && self.dec_layers == 4
                && self.num_queries == 100
                && self.out_feature_indexes == [3, 6, 9, 12]
                && self.positional_encoding_size == Some(32)
        } else {
            self.resolution == 560
                && self.patch_size == 14
                && self.num_windows == 4
                && self.dec_layers == 3
                && self.num_queries == 300
                && self.out_feature_indexes == [2, 5, 8, 11]
                && self.positional_encoding_size.is_none()
        };
        if !supported {
            return Err(if self.segmentation {
                "unsupported RF-DETR segmentation variant; expected released Seg Small geometry"
                    .into()
            } else {
                "unsupported RF-DETR detection variant; expected released Base geometry".into()
            });
        }
        self.validate_id2label()?;
        Ok(())
    }

    fn validate_id2label(&self) -> Result<(), String> {
        let Some(labels) = &self.id2label else {
            return Ok(());
        };
        let expected = self.logits_classes();
        if labels.len() != expected {
            return Err(format!(
                "RF-DETR id2label must contain exactly {expected} dense labels; got {}",
                labels.len()
            ));
        }
        for (key, label) in labels {
            let index = key.parse::<usize>().map_err(|_| {
                format!("RF-DETR id2label key {key:?} must be a canonical non-negative integer")
            })?;
            if key != &index.to_string() || index >= expected {
                return Err(format!(
                    "RF-DETR id2label key {key:?} must be in canonical range 0..{expected}"
                ));
            }
            if label.trim().is_empty() {
                return Err(format!("RF-DETR id2label[{index}] must not be empty"));
            }
        }
        for index in 0..expected {
            if !labels.contains_key(&index.to_string()) {
                return Err(format!("RF-DETR id2label is missing dense index {index}"));
            }
        }
        Ok(())
    }

    /// Zero-based backbone layers whose normed outputs feed the projector.
    #[must_use]
    pub fn feature_layers(&self) -> Vec<usize> {
        self.out_feature_indexes
            .iter()
            .map(|index| index.saturating_sub(1))
            .collect()
    }

    /// Legacy global-attention layer indexes used by the released graph.
    #[must_use]
    pub fn global_layers(&self) -> Vec<usize> {
        self.out_feature_indexes
            .iter()
            .copied()
            .filter(|&index| index < BACKBONE_LAYERS)
            .collect()
    }

    /// Raw classifier width, including RF-DETR's background logit.
    #[must_use]
    pub fn logits_classes(&self) -> usize {
        self.num_classes + 1
    }

    #[must_use]
    pub fn supports_masks(&self) -> bool {
        self.segmentation
    }

    #[must_use]
    pub fn default_threshold(&self) -> f32 {
        0.5
    }

    /// Custom label names in classifier-index order, when supplied by the checkpoint.
    #[must_use]
    pub fn class_names(&self) -> Option<Vec<String>> {
        let labels = self.id2label.as_ref()?;
        (0..self.logits_classes())
            .map(|index| labels.get(&index.to_string()).cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_released_base_and_derives_legacy_layer_sets() {
        let config = RfDetrConfig::from_json_str(
            r#"{"model_type":"rf-detr","encoder":"dinov2_windowed_small",
                "hidden_dim":256,"resolution":560,"dec_layers":3,"num_queries":300,
                "num_classes":90,"patch_size":14,"num_windows":4,"group_detr":13,
                "sa_nheads":8,"ca_nheads":16,"dec_n_points":2,"two_stage":true,
                "bbox_reparam":true,"lite_refpoint_refine":true,"layer_norm":true,
                "out_feature_indexes":[2,5,8,11],"projector_scale":["P4"]}"#,
        )
        .unwrap();
        assert_eq!(config.feature_layers(), [1, 4, 7, 10]);
        assert_eq!(config.global_layers(), [2, 5, 8, 11]);
        assert_eq!(config.logits_classes(), 91);
        assert!(!config.supports_masks());
        assert_eq!(config.default_threshold(), 0.5);
    }

    #[test]
    fn parses_released_seg_small_defaults() {
        let config = RfDetrConfig::from_json_str(
            r#"{"model_type":"rf-detr","resolution":384,"dec_layers":4,
                "num_queries":100,"patch_size":12,"num_windows":2,
                "out_feature_indexes":[3,6,9,12],"segmentation":true,
                "positional_encoding_size":32}"#,
        )
        .unwrap();
        assert_eq!(config.feature_layers(), [2, 5, 8, 11]);
        assert_eq!(config.global_layers(), [3, 6, 9]);
        assert!(config.supports_masks());
    }

    #[test]
    fn rejects_nearby_unreleased_or_resource_amplifying_variants() {
        let mut config = RfDetrConfig::default();
        config.num_queries = MAX_QUERIES + 1;
        assert!(config.validate().unwrap_err().contains("num_queries"));

        let mut config = RfDetrConfig::default();
        config.out_feature_indexes = vec![2, 5, 8, 12];
        assert!(config.validate().unwrap_err().contains("released Base"));

        let mut config = RfDetrConfig::default();
        config.segmentation = true;
        assert!(config.validate().unwrap_err().contains("Seg Small"));
    }

    #[test]
    fn parses_dense_custom_labels_in_classifier_order() {
        let labels: BTreeMap<_, _> = (0..91)
            .map(|index| (index.to_string(), format!("custom-{index}")))
            .collect();
        let json = serde_json::json!({
            "id2label": labels,
        });
        let config = RfDetrConfig::from_json_str(&json.to_string()).unwrap();
        let names = config.class_names().unwrap();
        assert_eq!(names.len(), 91);
        assert_eq!(names[0], "custom-0");
        assert_eq!(names[90], "custom-90");
    }

    #[test]
    fn rejects_malformed_sparse_or_empty_custom_labels() {
        let mut config = RfDetrConfig::default();
        config.id2label = Some(
            (0..91)
                .map(|index| (index.to_string(), format!("label-{index}")))
                .collect(),
        );
        config.id2label.as_mut().unwrap().remove("12");
        assert!(config.validate().unwrap_err().contains("exactly 91"));

        config
            .id2label
            .as_mut()
            .unwrap()
            .insert("12".into(), "ok".into());
        config.id2label.as_mut().unwrap().remove("7");
        config
            .id2label
            .as_mut()
            .unwrap()
            .insert("07".into(), "bad".into());
        assert!(config.validate().unwrap_err().contains("canonical"));

        config.id2label.as_mut().unwrap().remove("07");
        config
            .id2label
            .as_mut()
            .unwrap()
            .insert("7".into(), " ".into());
        assert!(config.validate().unwrap_err().contains("must not be empty"));
    }
}
