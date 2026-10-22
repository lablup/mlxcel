// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Published Sapiens2 config schemas.
//!
//! Early segmentation exports used nested `backbone_config` and legacy head
//! names. Later exports use the Transformers-style flat schema. Both are
//! canonicalized here so public checkpoints load without conversion.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

const MAX_HIDDEN_SIZE: usize = 2_432;
const MAX_HIDDEN_LAYERS: usize = 56;
const MAX_ATTENTION_HEADS: usize = 32;
const MAX_INTERMEDIATE_SIZE: usize = MAX_HIDDEN_SIZE * 4;
const MAX_REGISTER_TOKENS: usize = 8;
const MAX_LABELS: usize = 308;
const MAX_HEAD_STAGES: usize = 8;
const STANDARD_IMAGE_SIZE: [usize; 2] = [1_024, 768];
const STANDARD_PATCH_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sapiens2Task {
    Backbone,
    Pose,
    Segmentation,
    Normal,
    Pointmap,
    Matting,
}

impl Sapiens2Task {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Backbone => "backbone",
            Self::Pose => "pose",
            Self::Segmentation => "seg",
            Self::Normal => "normal",
            Self::Pointmap => "pointmap",
            Self::Matting => "matting",
        }
    }

    fn from_architecture(name: &str) -> Option<Self> {
        Some(match name {
            "Sapiens2Model" | "Sapiens2Backbone" => Self::Backbone,
            "Sapiens2ForPoseEstimation" => Self::Pose,
            "Sapiens2ForSemanticSegmentation" => Self::Segmentation,
            "Sapiens2ForNormalEstimation" => Self::Normal,
            "Sapiens2ForPointmapEstimation" => Self::Pointmap,
            "Sapiens2ForImageMatting" => Self::Matting,
            _ => return None,
        })
    }

    fn parse_legacy(name: &str) -> Option<Self> {
        Some(match name {
            "backbone" => Self::Backbone,
            "pose" => Self::Pose,
            "seg" => Self::Segmentation,
            "normal" => Self::Normal,
            "pointmap" => Self::Pointmap,
            "matting" => Self::Matting,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HeadConfig {
    pub upsample_out_channels: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub use_pixel_shuffle: bool,
    pub conv_out_channels: Vec<usize>,
    pub conv_kernel_sizes: Vec<usize>,
    pub scale_conv_out_channels: Vec<usize>,
    pub scale_conv_kernel_sizes: Vec<usize>,
    pub scale_final_input_size: Option<usize>,
    pub scale_final_hidden_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Sapiens2Config {
    pub task: Sapiens2Task,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub kv_heads_per_layer: Vec<usize>,
    pub intermediate_size: usize,
    /// `(height, width)`.
    pub image_size: (usize, usize),
    pub patch_size: usize,
    pub num_channels: usize,
    pub num_register_tokens: usize,
    pub rms_norm_eps: f32,
    pub normalize_backbone_outputs: bool,
    pub head: Option<HeadConfig>,
    pub num_labels: usize,
    pub keypoint_names: Option<Vec<String>>,
    pub flip_pairs: Vec<(usize, usize)>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    task: Option<String>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads_per_layer: Option<Vec<usize>>,
    num_first_full_attention_layers: Option<usize>,
    num_last_full_attention_layers: Option<usize>,
    intermediate_size: Option<usize>,
    image_size: Option<Vec<usize>>,
    patch_size: Option<usize>,
    num_channels: Option<usize>,
    num_register_tokens: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    normalize_backbone_outputs: Option<bool>,
    use_qk_norm: Option<bool>,
    query_bias: Option<bool>,
    key_bias: Option<bool>,
    value_bias: Option<bool>,
    proj_bias: Option<bool>,
    mlp_bias: Option<bool>,
    use_gated_mlp: Option<bool>,
    hidden_act: Option<String>,
    use_tokenizer: Option<bool>,
    head_config: Option<RawHeadConfig>,
    backbone_config: Option<LegacyBackboneConfig>,
    id2label: Option<BTreeMap<String, String>>,
    num_labels: Option<usize>,
    flip_pairs: Option<Vec<[usize; 2]>>,
}

#[derive(Debug, Deserialize)]
struct LegacyBackboneConfig {
    embed_dims: usize,
    num_layers: usize,
    num_heads: usize,
    feedforward_channels: usize,
    image_size: Vec<usize>,
    patch_size: usize,
    n_storage_tokens: usize,
    rope_base: Option<f32>,
    mhsa_early: Option<usize>,
    mhsa_late: Option<usize>,
    final_norm: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawHeadConfig {
    upsample_out_channels: Option<Vec<usize>>,
    upsample_kernel_sizes: Option<Vec<usize>>,
    use_pixel_shuffle: Option<bool>,
    conv_out_channels: Option<Vec<usize>>,
    conv_kernel_sizes: Option<Vec<usize>>,
    scale_conv_out_channels: Option<Vec<usize>>,
    scale_conv_kernel_sizes: Option<Vec<usize>>,
    scale_final_input_size: Option<usize>,
    scale_final_hidden_sizes: Option<Vec<usize>>,
    deconv_out_channels: Option<Vec<usize>>,
    deconv_kernel_sizes: Option<Vec<usize>>,
    upsample_channels: Option<Vec<usize>>,
    scale_final_layer: Option<Vec<usize>>,
    num_keypoints: Option<usize>,
    num_classes: Option<usize>,
}

impl Sapiens2Config {
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let raw: RawConfig = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", config_path.display()))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawConfig) -> Result<Self> {
        let task = raw
            .architectures
            .iter()
            .find_map(|name| Sapiens2Task::from_architecture(name))
            .or_else(|| raw.task.as_deref().and_then(Sapiens2Task::parse_legacy))
            .unwrap_or(Sapiens2Task::Backbone);
        let legacy = raw.backbone_config.as_ref();
        let hidden_size = raw
            .hidden_size
            .or_else(|| legacy.map(|v| v.embed_dims))
            .unwrap_or(1536);
        let num_hidden_layers = raw
            .num_hidden_layers
            .or_else(|| legacy.map(|v| v.num_layers))
            .unwrap_or(40);
        let num_attention_heads = raw
            .num_attention_heads
            .or_else(|| legacy.map(|v| v.num_heads))
            .unwrap_or(24);
        let default_intermediate = hidden_size
            .checked_mul(4)
            .context("Sapiens2 hidden size overflows its default intermediate size")?;
        let intermediate_size = raw
            .intermediate_size
            .or_else(|| legacy.map(|v| v.feedforward_channels))
            .unwrap_or(default_intermediate);
        let image_size = raw
            .image_size
            .or_else(|| legacy.map(|v| v.image_size.clone()))
            .unwrap_or_else(|| vec![1024, 768]);
        ensure!(
            image_size.len() == 2,
            "Sapiens2 image_size must be [height, width]"
        );
        let patch_size = raw
            .patch_size
            .or_else(|| legacy.map(|v| v.patch_size))
            .unwrap_or(16);
        let num_register_tokens = raw
            .num_register_tokens
            .or_else(|| legacy.map(|v| v.n_storage_tokens))
            .unwrap_or(8);
        let num_channels = raw.num_channels.unwrap_or(3);
        ensure!(
            hidden_size > 0
                && hidden_size <= MAX_HIDDEN_SIZE
                && num_hidden_layers > 0
                && num_hidden_layers <= MAX_HIDDEN_LAYERS
                && num_attention_heads > 0
                && num_attention_heads <= MAX_ATTENTION_HEADS
                && intermediate_size > 0
                && intermediate_size <= MAX_INTERMEDIATE_SIZE
                && num_register_tokens <= MAX_REGISTER_TOKENS,
            "Sapiens2 backbone dimensions exceed the published 5B architecture"
        );
        ensure!(
            image_size == STANDARD_IMAGE_SIZE
                && patch_size == STANDARD_PATCH_SIZE
                && num_channels == 3,
            "Sapiens2 supports the published 1024x768 RGB patch-16 checkpoints"
        );
        ensure!(
            hidden_size.is_multiple_of(num_attention_heads),
            "Sapiens2 hidden size must divide evenly into attention heads"
        );
        ensure!(
            (hidden_size / num_attention_heads).is_multiple_of(4),
            "Sapiens2 attention head dimension must be divisible by four for 2D RoPE"
        );
        let first_full = raw
            .num_first_full_attention_layers
            .or_else(|| legacy.and_then(|v| v.mhsa_early))
            .unwrap_or(8);
        let last_full = raw
            .num_last_full_attention_layers
            .or_else(|| legacy.and_then(|v| v.mhsa_late))
            .unwrap_or(8);
        ensure!(
            first_full <= num_hidden_layers && last_full <= num_hidden_layers,
            "Sapiens2 full-attention ranges exceed the layer count"
        );
        let kv_heads_per_layer = raw.num_key_value_heads_per_layer.unwrap_or_else(|| {
            (0..num_hidden_layers)
                .map(|index| {
                    if index < first_full || index >= num_hidden_layers.saturating_sub(last_full) {
                        num_attention_heads
                    } else {
                        num_attention_heads / 2
                    }
                })
                .collect()
        });
        ensure!(
            kv_heads_per_layer.len() == num_hidden_layers,
            "Sapiens2 KV-head list must contain one value per layer"
        );
        let patch_tokens = (image_size[0] / patch_size)
            .checked_mul(image_size[1] / patch_size)
            .context("Sapiens2 patch grid overflows")?;
        ensure!(
            patch_tokens == 3_072
                && patch_tokens
                    .checked_add(num_register_tokens + 1)
                    .is_some_and(|tokens| tokens <= 3_081),
            "Sapiens2 token grid exceeds the published 1024x768 checkpoint scope"
        );
        ensure!(
            kv_heads_per_layer
                .iter()
                .all(|&heads| heads > 0 && num_attention_heads.is_multiple_of(heads)),
            "Sapiens2 query heads must be divisible by every KV-head count"
        );
        ensure!(
            raw.use_qk_norm.unwrap_or(true),
            "Sapiens2 checkpoints without QK RMSNorm are not supported"
        );
        ensure!(
            raw.query_bias.unwrap_or(true)
                && raw.key_bias.unwrap_or(true)
                && raw.value_bias.unwrap_or(true)
                && raw.proj_bias.unwrap_or(true)
                && raw.mlp_bias.unwrap_or(true),
            "Sapiens2 checkpoints must use the published projection and MLP biases"
        );
        ensure!(
            raw.use_gated_mlp.unwrap_or(true)
                && raw.hidden_act.as_deref().unwrap_or("silu") == "silu",
            "Sapiens2 checkpoints must use the published SwiGLU feed-forward block"
        );
        ensure!(
            !raw.use_tokenizer.unwrap_or(false),
            "Sapiens2 4K tokenizer checkpoints are not supported"
        );
        let rms_norm_eps = raw.rms_norm_eps.unwrap_or(1e-6);
        let rope_theta = raw
            .rope_theta
            .or_else(|| legacy.and_then(|value| value.rope_base))
            .unwrap_or(100.0);
        ensure!(
            rms_norm_eps.is_finite()
                && rms_norm_eps > 0.0
                && rope_theta.is_finite()
                && rope_theta > 0.0,
            "Sapiens2 normalization epsilon and RoPE theta must be finite and positive"
        );

        let raw_head = raw.head_config;
        let legacy_num_labels = raw_head.as_ref().and_then(|head| match task {
            Sapiens2Task::Segmentation => head.num_classes,
            Sapiens2Task::Pose => head.num_keypoints,
            _ => None,
        });
        let num_labels = raw
            .num_labels
            .or_else(|| raw.id2label.as_ref().map(BTreeMap::len))
            .or(legacy_num_labels)
            .unwrap_or(match task {
                Sapiens2Task::Pose => 308,
                Sapiens2Task::Segmentation => 29,
                Sapiens2Task::Normal | Sapiens2Task::Pointmap => 3,
                Sapiens2Task::Matting => 4,
                Sapiens2Task::Backbone => 0,
            });
        ensure!(
            task == Sapiens2Task::Backbone || (num_labels > 0 && num_labels <= MAX_LABELS),
            "Sapiens2 task label count exceeds the published heads"
        );
        let head = raw_head.map(canonicalize_head);
        if task != Sapiens2Task::Backbone && head.is_none() {
            bail!("Sapiens2 task `{}` requires head_config", task.name());
        }
        if let Some(head) = &head {
            ensure!(
                [
                    head.upsample_out_channels.len(),
                    head.upsample_kernel_sizes.len(),
                    head.conv_out_channels.len(),
                    head.conv_kernel_sizes.len(),
                    head.scale_conv_out_channels.len(),
                    head.scale_conv_kernel_sizes.len(),
                    head.scale_final_hidden_sizes.len(),
                ]
                .into_iter()
                .all(|length| length <= MAX_HEAD_STAGES),
                "Sapiens2 head contains too many stages"
            );
            ensure!(
                head.upsample_out_channels
                    .iter()
                    .chain(&head.conv_out_channels)
                    .chain(&head.scale_conv_out_channels)
                    .chain(&head.scale_final_hidden_sizes)
                    .all(|&value| value > 0 && value <= MAX_INTERMEDIATE_SIZE)
                    && head
                        .scale_final_input_size
                        .is_none_or(|value| value > 0 && value <= MAX_INTERMEDIATE_SIZE)
                    && head
                        .upsample_kernel_sizes
                        .iter()
                        .chain(&head.conv_kernel_sizes)
                        .chain(&head.scale_conv_kernel_sizes)
                        .all(|&value| (1..=16).contains(&value)),
                "Sapiens2 head dimensions exceed the published checkpoint scope"
            );
        }
        let keypoint_names = if task == Sapiens2Task::Pose {
            raw.id2label.as_ref().map(|labels| {
                (0..num_labels)
                    .map(|index| {
                        labels
                            .get(&index.to_string())
                            .cloned()
                            .unwrap_or_else(|| index.to_string())
                    })
                    .collect()
            })
        } else {
            None
        };
        let flip_pairs = raw
            .flip_pairs
            .unwrap_or_default()
            .into_iter()
            .map(|pair| (pair[0], pair[1]))
            .collect::<Vec<_>>();
        ensure!(
            flip_pairs
                .iter()
                .all(|&(left, right)| left < num_labels && right < num_labels),
            "Sapiens2 flip pair is outside the label range"
        );
        Ok(Self {
            task,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            kv_heads_per_layer,
            intermediate_size,
            image_size: (image_size[0], image_size[1]),
            patch_size,
            num_channels,
            num_register_tokens,
            rms_norm_eps,
            normalize_backbone_outputs: raw
                .normalize_backbone_outputs
                .or_else(|| legacy.and_then(|v| v.final_norm))
                .unwrap_or(true),
            head,
            num_labels,
            keypoint_names,
            flip_pairs,
        })
    }
}

fn canonicalize_head(raw: RawHeadConfig) -> HeadConfig {
    let legacy_scale = raw.scale_final_layer.unwrap_or_default();
    let legacy_pixel_shuffle = raw.upsample_channels.is_some();
    HeadConfig {
        upsample_out_channels: raw
            .upsample_out_channels
            .or(raw.deconv_out_channels)
            .or(raw.upsample_channels)
            .unwrap_or_default(),
        upsample_kernel_sizes: raw
            .upsample_kernel_sizes
            .or(raw.deconv_kernel_sizes)
            .unwrap_or_default(),
        use_pixel_shuffle: raw.use_pixel_shuffle.unwrap_or(legacy_pixel_shuffle),
        conv_out_channels: raw.conv_out_channels.unwrap_or_default(),
        conv_kernel_sizes: raw.conv_kernel_sizes.unwrap_or_default(),
        scale_conv_out_channels: raw.scale_conv_out_channels.unwrap_or_default(),
        scale_conv_kernel_sizes: raw.scale_conv_kernel_sizes.unwrap_or_default(),
        scale_final_input_size: raw
            .scale_final_input_size
            .or_else(|| legacy_scale.first().copied()),
        scale_final_hidden_sizes: raw.scale_final_hidden_sizes.unwrap_or_else(|| {
            if legacy_scale.len() > 2 {
                legacy_scale[1..legacy_scale.len() - 1].to_vec()
            } else {
                Vec::new()
            }
        }),
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
