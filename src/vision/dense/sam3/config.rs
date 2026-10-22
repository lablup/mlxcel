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

//! Nested SAM 3 checkpoint configuration for `sam3_video` and `sam3`.

use serde::Deserialize;

macro_rules! config {
    ($name:ident { $($field:ident: $ty:ty = $default:expr),* $(,)? }) => {
        #[derive(Debug, Clone, Deserialize)]
        #[serde(default)]
        pub struct $name {
            $(pub $field: $ty),*
        }

        impl Default for $name {
            fn default() -> Self {
                Self { $($field: $default),* }
            }
        }
    };
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            mode: "affine".into(),
        }
    }
}

config!(VitConfig {
    model_type: String = "sam3_vit_model".into(),
    hidden_size: usize = 1024,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 16,
    intermediate_size: usize = 4736,
    hidden_act: String = "gelu".into(),
    image_size: usize = 1008,
    patch_size: usize = 14,
    num_channels: usize = 3,
    window_size: usize = 24,
    global_attn_indexes: Vec<usize> = vec![7, 15, 23, 31],
    qkv_bias: bool = true,
    rope_theta: f32 = 10_000.0,
    pretrain_image_size: usize = 336,
    layer_norm_eps: f32 = 1e-6,
    layer_scale_init_value: Option<f32> = None,
    hidden_dropout: f32 = 0.0,
    attention_dropout: f32 = 0.0,
});

config!(VisionEncoderConfig {
    model_type: String = "sam3_vision_model".into(),
    backbone_config: VitConfig = VitConfig::default(),
    fpn_hidden_size: usize = 256,
    fpn_kernel_size: usize = 2,
    fpn_stride: usize = 2,
    scale_factors: Vec<f32> = vec![4.0, 2.0, 1.0, 0.5],
    num_feature_levels: usize = 3,
    backbone_feature_sizes: Vec<Vec<usize>> = vec![vec![288, 288], vec![144, 144], vec![72, 72]],
    layer_norm_eps: f32 = 1e-6,
});

config!(TextEncoderConfig {
    model_type: String = "clip_text_model".into(),
    hidden_size: usize = 1024,
    num_hidden_layers: usize = 24,
    num_attention_heads: usize = 16,
    intermediate_size: usize = 4096,
    hidden_act: String = "gelu".into(),
    vocab_size: usize = 49_408,
    max_position_embeddings: usize = 32,
    projection_dim: usize = 512,
    layer_norm_eps: f32 = 1e-5,
    attention_dropout: f32 = 0.0,
    bos_token_id: i32 = 49_406,
    eos_token_id: i32 = 49_407,
    pad_token_id: i32 = 1,
});

config!(DetrEncoderConfig {
    model_type: String = "sam3_detr_encoder".into(),
    hidden_size: usize = 256,
    num_layers: usize = 6,
    num_attention_heads: usize = 8,
    intermediate_size: usize = 2048,
    hidden_act: String = "relu".into(),
    dropout: f32 = 0.1,
    layer_norm_eps: f32 = 1e-6,
});

config!(DetrDecoderConfig {
    model_type: String = "sam3_detr_decoder".into(),
    hidden_size: usize = 256,
    num_layers: usize = 6,
    num_attention_heads: usize = 8,
    num_queries: usize = 200,
    intermediate_size: usize = 2048,
    hidden_act: String = "relu".into(),
    dropout: f32 = 0.1,
    layer_norm_eps: f32 = 1e-6,
    box_rpb_mode: String = "log".into(),
    use_presence_token: bool = true,
});

config!(GeometryEncoderConfig {
    model_type: String = "sam3_geometry_encoder".into(),
    hidden_size: usize = 256,
    num_layers: usize = 3,
    num_attention_heads: usize = 8,
    intermediate_size: usize = 2048,
    hidden_act: String = "relu".into(),
    dropout: f32 = 0.1,
    roi_size: usize = 7,
    layer_norm_eps: f32 = 1e-6,
});

config!(DetectorMaskDecoderConfig {
    model_type: String = "sam3_mask_decoder".into(),
    hidden_size: usize = 256,
    num_attention_heads: usize = 8,
    num_upsampling_stages: usize = 3,
    dropout: f32 = 0.0,
    layer_norm_eps: f32 = 1e-6,
});

config!(DetectorConfig {
    model_type: String = "sam3".into(),
    vision_config: VisionEncoderConfig = VisionEncoderConfig::default(),
    text_config: TextEncoderConfig = TextEncoderConfig::default(),
    detr_encoder_config: DetrEncoderConfig = DetrEncoderConfig::default(),
    detr_decoder_config: DetrDecoderConfig = DetrDecoderConfig::default(),
    geometry_encoder_config: GeometryEncoderConfig = GeometryEncoderConfig::default(),
    mask_decoder_config: DetectorMaskDecoderConfig = DetectorMaskDecoderConfig::default(),
    initializer_range: f32 = 0.02,
});

config!(TrackerMaskDecoderConfig {
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
});

config!(PromptEncoderConfig {
    hidden_size: usize = 256,
    image_size: usize = 1008,
    patch_size: usize = 14,
    mask_input_channels: usize = 16,
    num_point_embeddings: usize = 4,
    hidden_act: String = "gelu".into(),
    scale: usize = 1,
    layer_norm_eps: f32 = 1e-6,
});

config!(TrackerConfig {
    model_type: String = "sam3_tracker_video".into(),
    image_size: usize = 1008,
    vision_config: VisionEncoderConfig = VisionEncoderConfig::default(),
    mask_decoder_config: TrackerMaskDecoderConfig = TrackerMaskDecoderConfig::default(),
    prompt_encoder_config: PromptEncoderConfig = PromptEncoderConfig::default(),
    memory_attention_hidden_size: usize = 256,
    memory_attention_num_layers: usize = 4,
    memory_attention_num_attention_heads: usize = 1,
    memory_attention_feed_forward_hidden_size: usize = 2048,
    memory_attention_feed_forward_hidden_act: String = "relu".into(),
    memory_attention_dropout: f32 = 0.1,
    memory_attention_rope_dropout: f32 = 0.1,
    memory_attention_rope_theta: f32 = 10_000.0,
    memory_attention_rope_feat_sizes: Vec<usize> = vec![72, 72],
    memory_attention_downsample_rate: usize = 1,
    memory_encoder_hidden_size: usize = 256,
    memory_encoder_output_channels: usize = 64,
    mask_downsampler_embed_dim: usize = 256,
    mask_downsampler_kernel_size: usize = 3,
    mask_downsampler_stride: usize = 2,
    mask_downsampler_padding: usize = 1,
    mask_downsampler_total_stride: usize = 16,
    mask_downsampler_hidden_act: String = "gelu".into(),
    memory_fuser_embed_dim: usize = 256,
    memory_fuser_kernel_size: usize = 7,
    memory_fuser_padding: usize = 3,
    memory_fuser_num_layers: usize = 2,
    memory_fuser_intermediate_dim: usize = 1024,
    memory_fuser_layer_scale_init_value: f32 = 1e-6,
    memory_fuser_hidden_act: String = "gelu".into(),
    num_maskmem: usize = 7,
    max_cond_frame_num: usize = 4,
    max_object_pointers_in_encoder: usize = 16,
    multimask_output_in_sam: bool = true,
    multimask_output_for_tracking: bool = true,
    multimask_min_pt_num: usize = 0,
    multimask_max_pt_num: usize = 1,
    sigmoid_bias_for_mem_enc: f32 = -10.0,
    sigmoid_scale_for_mem_enc: f32 = 20.0,
    enable_occlusion_spatial_embedding: bool = true,
    enable_temporal_pos_encoding_for_object_pointers: bool = true,
});

config!(Sam3Config {
    model_type: String = "sam3_video".into(),
    detector_config: DetectorConfig = DetectorConfig::default(),
    tracker_config: TrackerConfig = TrackerConfig::default(),
    text_config: Option<TextEncoderConfig> = None,
    vision_config: Option<VisionEncoderConfig> = None,
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
    init_trk_keep_alive: i32 = 30,
    max_trk_keep_alive: i32 = 30,
    min_trk_keep_alive: i32 = -1,
    hotstart_delay: usize = 15,
    hotstart_dup_thresh: usize = 8,
    hotstart_unmatch_thresh: usize = 8,
    recondition_every_nth_frame: usize = 16,
    recondition_on_trk_masks: bool = false,
    decrease_trk_keep_alive_for_empty_masklets: bool = false,
    suppress_unmatched_only_within_hotstart: bool = true,
    suppress_overlapping_based_on_recent_occlusion_threshold: f32 = 0.7,
});

impl Sam3Config {
    #[must_use]
    pub fn quantization_params(&self) -> (i32, i32) {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
            .map_or((64, 4), |q| (q.group_size, q.bits))
    }

    /// Validate shape-critical settings before any weights are constructed.
    pub fn validate(&self) -> Result<(), String> {
        let det = &self.detector_config;
        validate_heads(
            "vision",
            det.vision_config.backbone_config.hidden_size,
            det.vision_config.backbone_config.num_attention_heads,
        )?;
        validate_heads(
            "text",
            det.text_config.hidden_size,
            det.text_config.num_attention_heads,
        )?;
        validate_heads(
            "DETR encoder",
            det.detr_encoder_config.hidden_size,
            det.detr_encoder_config.num_attention_heads,
        )?;
        validate_heads(
            "DETR decoder",
            det.detr_decoder_config.hidden_size,
            det.detr_decoder_config.num_attention_heads,
        )?;
        validate_heads(
            "geometry",
            det.geometry_encoder_config.hidden_size,
            det.geometry_encoder_config.num_attention_heads,
        )?;
        let tracker = &self.tracker_config;
        validate_heads(
            "tracker mask decoder",
            tracker.mask_decoder_config.hidden_size,
            tracker.mask_decoder_config.num_attention_heads,
        )?;
        validate_heads(
            "tracker memory attention",
            tracker.memory_attention_hidden_size,
            tracker.memory_attention_num_attention_heads,
        )?;
        let vit = &det.vision_config.backbone_config;
        if vit.patch_size == 0 || vit.image_size % vit.patch_size != 0 {
            return Err("SAM3 image_size must be divisible by patch_size".into());
        }
        if det.text_config.max_position_embeddings == 0 {
            return Err("SAM3 text max_position_embeddings must be positive".into());
        }
        if det.geometry_encoder_config.roi_size == 0 {
            return Err("SAM3 geometry roi_size must be positive".into());
        }
        let (group_size, bits) = self.quantization_params();
        if (self.quantization.is_some() || self.quantization_config.is_some())
            && (group_size <= 0 || bits <= 0)
        {
            return Err("SAM3 quantization group_size and bits must be positive".into());
        }
        Ok(())
    }
}

fn validate_heads(name: &str, hidden: usize, heads: usize) -> Result<(), String> {
    if heads == 0 || hidden == 0 || !hidden.is_multiple_of(heads) {
        Err(format!(
            "SAM3 {name} hidden size {hidden} is not divisible by {heads} heads"
        ))
    } else {
        Ok(())
    }
}
