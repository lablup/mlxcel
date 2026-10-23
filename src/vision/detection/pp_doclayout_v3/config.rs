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

//! PP-DocLayoutV3 configuration.

use std::collections::BTreeMap;

use serde::Deserialize;

/// HGNetV2-L backbone geometry used by PP-DocLayoutV3.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HgNetV2Config {
    pub model_type: String,
    pub arch: String,
    pub depths: Vec<usize>,
    pub hidden_sizes: Vec<usize>,
    pub embedding_size: usize,
    pub num_channels: usize,
    pub hidden_act: String,
    pub stem_channels: Vec<usize>,
    pub stem_strides: Vec<usize>,
    pub stage_in_channels: Vec<usize>,
    pub stage_mid_channels: Vec<usize>,
    pub stage_out_channels: Vec<usize>,
    pub stage_num_blocks: Vec<usize>,
    pub stage_numb_of_layers: Vec<usize>,
    pub stage_downsample: Vec<bool>,
    pub stage_downsample_strides: Vec<usize>,
    pub stage_light_block: Vec<bool>,
    pub stage_kernel_size: Vec<usize>,
    pub use_learnable_affine_block: bool,
    pub out_features: Vec<String>,
    /// Transformers calls this field `return_idx`; mlx-vlm calls it
    /// `out_indices`. The graph always returns all four stages.
    #[serde(alias = "return_idx")]
    pub out_indices: Vec<usize>,
}

impl Default for HgNetV2Config {
    fn default() -> Self {
        Self {
            model_type: "hgnet_v2".into(),
            arch: "L".into(),
            depths: vec![3, 4, 6, 3],
            hidden_sizes: vec![256, 512, 1024, 2048],
            embedding_size: 64,
            num_channels: 3,
            hidden_act: "relu".into(),
            stem_channels: vec![3, 32, 48],
            stem_strides: vec![2, 1, 1, 2, 1],
            stage_in_channels: vec![48, 128, 512, 1024],
            stage_mid_channels: vec![48, 96, 192, 384],
            stage_out_channels: vec![128, 512, 1024, 2048],
            stage_num_blocks: vec![1, 1, 3, 1],
            stage_numb_of_layers: vec![6, 6, 6, 6],
            stage_downsample: vec![false, true, true, true],
            stage_downsample_strides: vec![2, 2, 2, 2],
            stage_light_block: vec![false, false, true, true],
            stage_kernel_size: vec![3, 3, 5, 5],
            use_learnable_affine_block: false,
            out_features: ["stage1", "stage2", "stage3", "stage4"]
                .map(str::to_string)
                .to_vec(),
            out_indices: vec![1, 2, 3, 4],
        }
    }
}

impl HgNetV2Config {
    fn validate(&self) -> Result<(), String> {
        if self.model_type != "hgnet_v2" || self.arch != "L" {
            return Err(format!(
                "unsupported PP-DocLayoutV3 backbone: model_type={:?}, arch={:?}",
                self.model_type, self.arch
            ));
        }
        if self.num_channels != 3 || self.hidden_act != "relu" {
            return Err("PP-DocLayoutV3 requires a three-channel ReLU HGNetV2 backbone".into());
        }
        if self.depths != [3, 4, 6, 3]
            || self.hidden_sizes != [256, 512, 1024, 2048]
            || self.embedding_size != 64
            || self.use_learnable_affine_block
        {
            return Err("the PP-DocLayoutV3 graph supports only the stock HGNetV2-L body".into());
        }
        let stage_lengths = [
            self.stage_in_channels.len(),
            self.stage_mid_channels.len(),
            self.stage_out_channels.len(),
            self.stage_num_blocks.len(),
            self.stage_numb_of_layers.len(),
            self.stage_downsample.len(),
            self.stage_downsample_strides.len(),
            self.stage_light_block.len(),
            self.stage_kernel_size.len(),
        ];
        if stage_lengths.iter().any(|&length| length != 4) {
            return Err(format!(
                "HGNetV2 stage vectors must each contain four entries; lengths={stage_lengths:?}"
            ));
        }
        if self.stem_channels != [3, 32, 48] || self.stem_strides != [2, 1, 1, 2, 1] {
            return Err("the PP-DocLayoutV3 graph supports only the stock HGNetV2-L stem".into());
        }
        if self.stage_in_channels != [48, 128, 512, 1024]
            || self.stage_mid_channels != [48, 96, 192, 384]
            || self.stage_out_channels != [128, 512, 1024, 2048]
            || self.stage_num_blocks != [1, 1, 3, 1]
            || self.stage_numb_of_layers != [6, 6, 6, 6]
            || self.stage_downsample != [false, true, true, true]
            || self.stage_downsample_strides != [2, 2, 2, 2]
            || self.stage_light_block != [false, false, true, true]
            || self.stage_kernel_size != [3, 3, 5, 5]
        {
            return Err("the PP-DocLayoutV3 graph supports only the stock HGNetV2-L stages".into());
        }
        let expected_features = ["stage1", "stage2", "stage3", "stage4"];
        if self
            .out_features
            .iter()
            .map(String::as_str)
            .ne(expected_features)
            || (self.out_indices != [0, 1, 2, 3] && self.out_indices != [1, 2, 3, 4])
        {
            return Err("PP-DocLayoutV3 requires all four HGNetV2 output stages in order".into());
        }
        Ok(())
    }
}

/// Flat HuggingFace/mlx-vlm PP-DocLayoutV3 configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PpDocLayoutV3Config {
    pub model_type: String,
    pub image_size: usize,
    pub num_labels: usize,
    pub id2label: Option<BTreeMap<String, String>>,
    pub backbone_config: HgNetV2Config,
    pub d_model: usize,
    pub encoder_hidden_dim: usize,
    pub encoder_in_channels: Vec<usize>,
    #[serde(alias = "feature_strides")]
    pub feat_strides: Vec<usize>,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_attention_heads: usize,
    pub encoder_activation_function: String,
    pub encode_proj_layers: Vec<usize>,
    pub positional_encoding_temperature: f32,
    pub activation_function: String,
    pub normalize_before: bool,
    pub layer_norm_eps: f32,
    pub hidden_expansion: f32,
    pub batch_norm_eps: f32,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_in_channels: Vec<usize>,
    pub decoder_activation_function: String,
    pub num_feature_levels: usize,
    pub decoder_n_points: usize,
    pub num_queries: usize,
    pub learn_initial_query: bool,
    pub use_focal_loss: bool,
    pub global_pointer_head_size: usize,
    pub gp_dropout_value: f32,
    pub mask_enhanced: bool,
    pub mask_feature_channels: Vec<usize>,
    pub num_prototypes: usize,
    pub x4_feat_dim: usize,
}

impl Default for PpDocLayoutV3Config {
    fn default() -> Self {
        Self {
            model_type: "pp_doclayout_v3".into(),
            image_size: 800,
            num_labels: 25,
            id2label: None,
            backbone_config: HgNetV2Config::default(),
            d_model: 256,
            encoder_hidden_dim: 256,
            encoder_in_channels: vec![512, 1024, 2048],
            feat_strides: vec![8, 16, 32],
            encoder_layers: 1,
            encoder_ffn_dim: 1024,
            encoder_attention_heads: 8,
            encoder_activation_function: "gelu".into(),
            encode_proj_layers: vec![2],
            positional_encoding_temperature: 10000.0,
            activation_function: "silu".into(),
            normalize_before: false,
            layer_norm_eps: 1e-5,
            hidden_expansion: 1.0,
            batch_norm_eps: 1e-5,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 1024,
            decoder_in_channels: vec![256, 256, 256],
            decoder_activation_function: "relu".into(),
            num_feature_levels: 3,
            decoder_n_points: 4,
            num_queries: 300,
            learn_initial_query: false,
            use_focal_loss: true,
            global_pointer_head_size: 64,
            gp_dropout_value: 0.1,
            mask_enhanced: true,
            mask_feature_channels: vec![64, 64],
            num_prototypes: 32,
            x4_feat_dim: 128,
        }
    }
}

impl PpDocLayoutV3Config {
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let mut config: Self = serde_json::from_str(json)
            .map_err(|error| format!("PP-DocLayoutV3 config parse error: {error}"))?;
        if let Some(labels) = &config.id2label
            && !labels.is_empty()
        {
            config.num_labels = labels.len();
        }
        config.validate()?;
        Ok(config)
    }

    /// Reject configurations the inference graph cannot execute faithfully.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "pp_doclayout_v3" {
            return Err(format!("unsupported model_type {:?}", self.model_type));
        }
        self.backbone_config.validate()?;
        if self.image_size == 0 || self.num_labels == 0 || self.num_queries == 0 {
            return Err("image_size, num_labels, and num_queries must be positive".into());
        }
        if self.decoder_attention_heads == 0
            || !self.d_model.is_multiple_of(self.decoder_attention_heads)
        {
            return Err("d_model must be divisible by a positive decoder_attention_heads".into());
        }
        if self.encoder_attention_heads == 0
            || !self
                .encoder_hidden_dim
                .is_multiple_of(self.encoder_attention_heads)
            || !self.encoder_hidden_dim.is_multiple_of(4)
        {
            return Err(
                "encoder_hidden_dim must be divisible by encoder_attention_heads and by 4".into(),
            );
        }
        if self.encoder_hidden_dim != self.d_model {
            return Err("encoder_hidden_dim must equal d_model for the feature pyramid".into());
        }
        if self.encoder_in_channels.len() != self.num_feature_levels
            || self.decoder_in_channels.len() != self.num_feature_levels
            || self.feat_strides.len() != self.num_feature_levels
            || self.num_feature_levels == 0
        {
            return Err(
                "encoder, decoder, and stride vectors must match num_feature_levels".into(),
            );
        }
        if self.encoder_in_channels != self.backbone_config.stage_out_channels[1..] {
            return Err("encoder_in_channels must match HGNetV2 stages 2 through 4".into());
        }
        if self.feat_strides != [8, 16, 32] {
            return Err("the mask feature path requires feature strides [8, 16, 32]".into());
        }
        if self
            .decoder_in_channels
            .iter()
            .any(|&channels| channels != self.d_model)
        {
            return Err("every decoder input channel count must equal d_model".into());
        }
        if self.encode_proj_layers.len() != 1
            || self.encode_proj_layers[0] >= self.num_feature_levels
        {
            return Err("encode_proj_layers must select exactly one valid feature level".into());
        }
        if self.encoder_layers == 0
            || self.encoder_ffn_dim == 0
            || self.decoder_layers == 0
            || self.decoder_ffn_dim == 0
            || self.decoder_n_points == 0
            || self.global_pointer_head_size == 0
            || self.num_prototypes == 0
        {
            return Err(
                "layer, feed-forward, point, head, and prototype counts must be positive".into(),
            );
        }
        if !matches!(self.activation_function.as_str(), "relu" | "silu" | "gelu")
            || !matches!(
                self.encoder_activation_function.as_str(),
                "relu" | "silu" | "gelu"
            )
            || !matches!(
                self.decoder_activation_function.as_str(),
                "relu" | "silu" | "gelu"
            )
        {
            return Err("unsupported activation in PP-DocLayoutV3 configuration".into());
        }
        for (name, value) in [
            ("layer_norm_eps", self.layer_norm_eps),
            ("batch_norm_eps", self.batch_norm_eps),
            ("hidden_expansion", self.hidden_expansion),
            (
                "positional_encoding_temperature",
                self.positional_encoding_temperature,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if !self.gp_dropout_value.is_finite() || !(0.0..=1.0).contains(&self.gp_dropout_value) {
            return Err("gp_dropout_value must be finite and in [0, 1]".into());
        }
        if self.normalize_before || self.learn_initial_query || !self.use_focal_loss {
            return Err(
                "only post-norm, encoder-selected queries with focal classification are supported"
                    .into(),
            );
        }
        if !self.mask_enhanced
            || self.mask_feature_channels.len() != 2
            || self.mask_feature_channels != [64, 64]
            || self.num_prototypes != 32
            || self.x4_feat_dim != self.backbone_config.stage_out_channels[0]
        {
            return Err("invalid mask-enhanced query initialization geometry".into());
        }
        super::config_limits::validate(self)?;
        if let Some(labels) = &self.id2label
            && !labels.is_empty()
        {
            for id in 0..self.num_labels {
                if !labels.contains_key(&id.to_string()) {
                    return Err(format!("id2label is missing class id {id}"));
                }
            }
        }
        Ok(())
    }

    pub fn class_names(&self) -> Option<Vec<String>> {
        let labels = self.id2label.as_ref()?;
        if labels.is_empty() {
            return None;
        }
        (0..self.num_labels)
            .map(|id| labels.get(&id.to_string()).cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pinned_huggingface_aliases_and_infers_labels() {
        let config = PpDocLayoutV3Config::from_json_str(
            r#"{
                "model_type":"pp_doclayout_v3",
                "feature_strides":[8,16,32],
                "id2label":{"0":"title","1":"text"},
                "backbone_config":{"model_type":"hgnet_v2","arch":"L",
                    "return_idx":[0,1,2,3],
                    "out_features":["stage1","stage2","stage3","stage4"]}
            }"#,
        )
        .unwrap();
        assert_eq!(config.image_size, 800);
        assert_eq!(config.num_labels, 2);
        assert_eq!(config.feat_strides, [8, 16, 32]);
        assert_eq!(config.backbone_config.out_indices, [0, 1, 2, 3]);
        assert_eq!(config.class_names().unwrap(), ["title", "text"]);
    }

    #[test]
    fn validation_rejects_noncontiguous_label_map() {
        let error = PpDocLayoutV3Config::from_json_str(
            r#"{"model_type":"pp_doclayout_v3","id2label":{"0":"title","2":"text"}}"#,
        )
        .unwrap_err();
        assert!(error.contains("missing class id 1"));
    }

    #[test]
    fn validation_rejects_incompatible_attention_geometry() {
        let mut config = PpDocLayoutV3Config::default();
        config.decoder_attention_heads = 7;
        assert!(config.validate().unwrap_err().contains("d_model"));
        let mut oversized = PpDocLayoutV3Config::default();
        oversized.num_queries = 513;
        assert!(
            oversized
                .validate()
                .unwrap_err()
                .contains("inference limits")
        );
    }

    #[test]
    fn validation_pins_the_published_hgnet_stage_layout() {
        let mut config = PpDocLayoutV3Config::default();
        config.backbone_config.stage_kernel_size[0] = 5;
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("stock HGNetV2-L stages")
        );

        let mut config = PpDocLayoutV3Config::default();
        config.backbone_config.stage_num_blocks[2] = 4;
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("stock HGNetV2-L stages")
        );
    }
}
