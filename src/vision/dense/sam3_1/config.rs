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

//! Shape-critical SAM 3.1 checkpoint configuration.

use serde::Deserialize;

use super::super::sam3::config::{
    DetectorConfig, PromptEncoderConfig, QuantizationConfig, VisionEncoderConfig,
    validate_detector_config, validate_quantization_configs,
};

macro_rules! config {
    ($name:ident { $($field:ident: $ty:ty = $default:expr),* $(,)? }) => {
        #[derive(Debug, Clone, Deserialize)]
        #[serde(default)]
        pub(crate) struct $name {
            $(pub $field: $ty),*
        }

        impl Default for $name {
            fn default() -> Self {
                Self { $($field: $default),* }
            }
        }
    };
}

config!(MultiplexMaskDecoderConfig {
    hidden_size: usize = 256,
    num_hidden_layers: usize = 2,
    num_attention_heads: usize = 8,
    attention_downsample_rate: usize = 2,
    num_multimask_outputs: usize = 3,
    iou_head_depth: usize = 3,
    iou_head_hidden_dim: usize = 256,
    mlp_dim: usize = 2048,
    hidden_act: String = "gelu".into(),
    dynamic_multimask_via_stability: bool = true,
    dynamic_multimask_stability_delta: f32 = 0.05,
    dynamic_multimask_stability_thresh: f32 = 0.98,
    multiplex_count: usize = 16,
    multimask_outputs_only: bool = true,
    use_multimask_token_for_obj_ptr: bool = true,
});

config!(TrackerConfig {
    model_type: String = "sam3.1_tracker_video".into(),
    image_size: usize = 1008,
    vision_config: VisionEncoderConfig = sam31_vision_config(),
    mask_decoder_config: MultiplexMaskDecoderConfig = MultiplexMaskDecoderConfig::default(),
    prompt_encoder_config: PromptEncoderConfig = PromptEncoderConfig::default(),
    multiplex_count: usize = 16,
    memory_attention_feed_forward_hidden_size: usize = 2048,
    memory_attention_hidden_size: usize = 256,
    memory_attention_num_attention_heads: usize = 8,
    memory_attention_num_layers: usize = 4,
    memory_attention_rope_feat_sizes: Vec<usize> = vec![72, 72],
    memory_attention_rope_theta: f32 = 10_000.0,
    mask_downsampler_embed_dim: usize = 256,
    mask_downsampler_first_channels: usize = 16,
    mask_downsampler_input_size: usize = 1152,
    mask_downsampler_kernel_size: usize = 3,
    mask_downsampler_padding: usize = 1,
    mask_downsampler_stride: usize = 2,
    memory_encoder_hidden_size: usize = 256,
    memory_fuser_embed_dim: usize = 256,
    memory_fuser_intermediate_dim: usize = 1024,
    memory_fuser_kernel_size: usize = 7,
    memory_fuser_layer_scale_init_value: f32 = 1e-6,
    memory_fuser_num_layers: usize = 2,
    memory_fuser_padding: usize = 3,
    max_cond_frame_num: usize = 4,
    max_object_pointers_in_encoder: usize = 16,
    memory_temporal_stride_for_eval: usize = 1,
    num_maskmem: usize = 7,
    save_image_features: bool = true,
    use_maskmem_tpos_v2: bool = true,
    apply_sigmoid_to_mask_logits_for_mem_enc: bool = true,
    sigmoid_bias_for_mem_enc: f32 = -1.0,
    sigmoid_scale_for_mem_enc: f32 = 2.0,
    condition_as_mask_input: bool = true,
    condition_as_mask_input_bg: f32 = 0.0,
    condition_as_mask_input_fg: f32 = 1.0,
    directly_add_no_mem_embed: bool = true,
    multimask_max_pt_num: usize = 1,
    multimask_min_pt_num: usize = 0,
    multimask_output_for_tracking: bool = true,
    multimask_output_in_sam: bool = true,
    num_multimask_outputs: usize = 3,
    use_mask_input_as_output_without_sam: bool = true,
    add_output_suppression_embeddings: bool = true,
    fixed_no_obj_ptr: bool = true,
    object_score_logit_threshold: f32 = 0.0,
    pred_obj_scores: bool = true,
    use_linear_no_obj_ptr: bool = true,
    use_no_obj_ptr: bool = true,
    use_obj_ptrs_in_encoder: bool = true,
});

config!(Sam31Config {
    model_type: String = "sam3.1_video".into(),
    detector_config: DetectorConfig = sam31_detector_config(),
    tracker_config: TrackerConfig = TrackerConfig::default(),
    quantization: Option<QuantizationConfig> = None,
    quantization_config: Option<QuantizationConfig> = None,
    initializer_range: f32 = 0.02,
    low_res_mask_size: usize = 288,
    det_nms_thresh: f32 = 0.1,
    assoc_iou_thresh: f32 = 0.1,
    trk_assoc_iou_thresh: f32 = 0.5,
    high_conf_thresh: f32 = 0.8,
    high_iou_thresh: f32 = 0.8,
    new_det_thresh: f32 = 0.7,
    score_threshold_detection: f32 = 0.5,
    fill_hole_area: usize = 16,
    max_num_objects: usize = 10_000,
});

fn sam31_vision_config() -> VisionEncoderConfig {
    VisionEncoderConfig {
        scale_factors: vec![4.0, 2.0, 1.0],
        num_feature_levels: 3,
        ..VisionEncoderConfig::default()
    }
}

fn sam31_detector_config() -> DetectorConfig {
    DetectorConfig {
        model_type: "sam3.1".into(),
        vision_config: sam31_vision_config(),
        ..DetectorConfig::default()
    }
}

impl TrackerConfig {
    /// The published checkpoint retains a SAM 3 tracker marker and several
    /// stale scalar values. The pinned SAM 3.1 constructor replaces them with
    /// multiplex architecture constants before constructing any layer.
    #[must_use]
    pub(crate) fn normalize(mut self) -> Self {
        if self.model_type == "sam3_tracker_video" {
            self.model_type = "sam3.1_tracker_video".into();
            self.memory_attention_num_attention_heads = 8;
            self.sigmoid_scale_for_mem_enc = 2.0;
            self.sigmoid_bias_for_mem_enc = -1.0;
        }
        self.vision_config = sam31_vision_config();
        self.mask_decoder_config.multiplex_count = self.multiplex_count;
        self.mask_decoder_config.multimask_outputs_only = true;
        self
    }
}

impl Sam31Config {
    #[must_use]
    pub(crate) fn normalize(mut self) -> Self {
        self.tracker_config = self.tracker_config.normalize();
        self.detector_config.model_type = "sam3.1".into();
        self.detector_config.vision_config.scale_factors = vec![4.0, 2.0, 1.0];
        self.detector_config.vision_config.num_feature_levels = 3;
        self
    }

    #[must_use]
    pub(crate) fn quantization_params(&self) -> (i32, i32) {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
            .map_or((64, 4), |q| (q.group_size, q.bits))
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if !matches!(self.model_type.as_str(), "sam3.1_video" | "sam3.1") {
            return Err(format!(
                "unsupported SAM 3.1 model_type {:?}",
                self.model_type
            ));
        }
        let tracker = &self.tracker_config;
        validate_detector_config(&self.detector_config)?;
        if tracker.model_type != "sam3.1_tracker_video" {
            return Err(format!(
                "unsupported SAM 3.1 tracker model_type {:?}",
                tracker.model_type
            ));
        }
        let decoder = &tracker.mask_decoder_config;
        if tracker.multiplex_count != 16
            || decoder.multiplex_count != 16
            || decoder.hidden_size != 256
            || decoder.num_attention_heads != 8
            || decoder.attention_downsample_rate != 2
            || decoder.num_hidden_layers != 2
            || decoder.num_multimask_outputs != 3
            || decoder.iou_head_depth != 3
            || decoder.iou_head_hidden_dim != 256
            || decoder.mlp_dim != 2048
        {
            return Err("SAM 3.1 requires the published 16-way multiplex decoder shape".into());
        }
        if tracker.memory_attention_hidden_size != 256
            || tracker.memory_attention_num_attention_heads != 8
            || tracker.memory_attention_num_layers != 4
            || tracker.memory_attention_feed_forward_hidden_size != 2048
            || !tracker
                .memory_attention_hidden_size
                .is_multiple_of(tracker.memory_attention_num_attention_heads)
            || !(tracker.memory_attention_hidden_size
                / tracker.memory_attention_num_attention_heads)
                .is_multiple_of(4)
        {
            return Err("SAM 3.1 requires the published decoupled-memory attention shape".into());
        }
        if tracker.memory_attention_rope_feat_sizes.as_slice() != [72, 72]
            || !tracker.memory_attention_rope_theta.is_finite()
            || tracker.memory_attention_rope_theta <= 0.0
        {
            return Err("SAM 3.1 requires finite 72x72 memory RoPE settings".into());
        }
        if tracker.memory_temporal_stride_for_eval != 1
            || tracker.num_maskmem != 7
            || tracker.max_cond_frame_num != 4
            || tracker.max_object_pointers_in_encoder != 16
        {
            return Err("SAM 3.1 requires the published bounded memory history".into());
        }
        if tracker.image_size != 1008
            || tracker.mask_downsampler_input_size != 1152
            || tracker.mask_downsampler_embed_dim != 256
            || tracker.mask_downsampler_first_channels != 16
            || tracker.mask_downsampler_kernel_size != 3
            || tracker.mask_downsampler_padding != 1
            || tracker.mask_downsampler_stride != 2
            || tracker.memory_encoder_hidden_size != 256
            || tracker.memory_fuser_embed_dim != 256
            || tracker.memory_fuser_intermediate_dim != 1024
            || tracker.memory_fuser_kernel_size != 7
            || tracker.memory_fuser_num_layers != 2
            || tracker.memory_fuser_padding != 3
            || tracker.vision_config.backbone_config.patch_size == 0
            || !tracker
                .image_size
                .is_multiple_of(tracker.vision_config.backbone_config.patch_size)
        {
            return Err(
                "SAM 3.1 requires the published tracker image and downsampler sizes".into(),
            );
        }
        if self.max_num_objects == 0 || self.max_num_objects > 10_000 {
            return Err("SAM 3.1 max_num_objects must be between 1 and 10000".into());
        }
        validate_quantization_configs(
            self.quantization.as_ref(),
            self.quantization_config.as_ref(),
            "SAM 3.1",
        )?;
        if self.detector_config.vision_config.scale_factors != [4.0, 2.0, 1.0] {
            return Err("SAM 3.1 detector requires three 4x/2x/1x FPN levels".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_tracker_fields_are_replaced_by_sam31_constants() {
        let config: Sam31Config = serde_json::from_value(serde_json::json!({
            "model_type": "sam3.1_video",
            "tracker_config": {
                "model_type": "sam3_tracker_video",
                "memory_attention_num_attention_heads": 1,
                "sigmoid_scale_for_mem_enc": 20.0,
                "sigmoid_bias_for_mem_enc": -10.0
            }
        }))
        .unwrap();
        let tracker = config.normalize().tracker_config;
        assert_eq!(tracker.memory_attention_num_attention_heads, 8);
        assert_eq!(tracker.sigmoid_scale_for_mem_enc, 2.0);
        assert_eq!(tracker.sigmoid_bias_for_mem_enc, -1.0);
        assert!(tracker.mask_decoder_config.multimask_outputs_only);
    }

    #[test]
    fn malformed_memory_attention_config_is_rejected_without_panicking() {
        let mut config = Sam31Config::default().normalize();
        config.tracker_config.memory_attention_num_attention_heads = 0;
        assert!(config.validate().is_err());

        let mut config = Sam31Config::default().normalize();
        config.tracker_config.memory_attention_rope_feat_sizes = vec![72];
        assert!(config.validate().is_err());

        let mut config = Sam31Config::default().normalize();
        config.tracker_config.memory_temporal_stride_for_eval = 0;
        assert!(config.validate().is_err());

        let mut config = Sam31Config::default().normalize();
        config
            .tracker_config
            .mask_decoder_config
            .attention_downsample_rate = 0;
        assert!(config.validate().is_err());

        let mut config = Sam31Config::default().normalize();
        config.tracker_config.memory_fuser_num_layers = usize::MAX;
        assert!(config.validate().is_err());

        let mut config = Sam31Config::default().normalize();
        config
            .detector_config
            .detr_encoder_config
            .num_attention_heads = 0;
        assert!(config.validate().is_err());

        let mut config = Sam31Config::default().normalize();
        config.quantization = Some(QuantizationConfig {
            group_size: 0,
            bits: 4,
            mode: "affine".into(),
        });
        assert!(config.validate().is_err());
    }
}
