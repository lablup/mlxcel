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

//! Florence-2 DaViT vision backbone (`vision_tower.*`).
//!
//! DaViT ("Dual Attention Vision Transformer") is a four-stage hierarchical
//! encoder. Each stage starts with a strided `ConvEmbed` patch embedding that
//! halves (or quarters, at stage 0) the spatial extent and widens the channel
//! dimension, then runs `depths[i]` blocks that alternate windowed spatial
//! attention with channel attention.
//!
//! For the `Florence-2-base-ft` geometry and a 768x768 input the stage
//! progression is 192x192 @ 128 -> 96x96 @ 256 -> 48x48 @ 512 -> 24x24 @ 1024,
//! so the backbone emits `[B, 576, 1024]`.
//!
//! Scope: this module is the backbone only. The 1024 -> 768 `image_projection`
//! / `image_proj_norm`, the learned 2-D position embedding, the temporal
//! embedding, and the `image_feature_source` pooling all belong to the fusion
//! stage, which reads the config fields retained on
//! [`Florence2VisionConfig`] and consumes the tensors this module
//! deliberately leaves untouched.
//!
//! Reference: mlx-vlm `mlx_vlm/models/florence2/vision.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py>).

use std::path::Path;

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::florence2::Florence2Quantization;

use super::florence2_davit_blocks::{Block, BlockParams, ConvEmbed};
use super::{VisionEncoder, VisionEncoderOutput};

/// Weight-key prefix of the DaViT tower inside a full Florence-2 checkpoint.
pub const FLORENCE2_VISION_PREFIX: &str = "vision_tower.";

/// Upper bound accepted for `depths[i]`. Real Florence-2 towers use at most 9
/// blocks in a stage. The cap exists because [`Florence2DaViT::from_weights`]
/// sizes a `Vec` from this field before it looks up a single weight, so an
/// absurd value out of a hostile `config.json` would turn into an
/// allocation-failure abort rather than an error return.
const MAX_STAGE_DEPTH: i32 = 1024;

/// DaViT geometry parsed from the `vision_config` object of a Florence-2
/// `config.json`.
///
/// The trailing four fields (`projection_dim`, `image_pos_embed`,
/// `visual_temporal_embedding`, `image_feature_source`) are not used by the
/// backbone forward. Upstream carries them on the *model* config rather than
/// the vision config, but real Florence-2 checkpoints put them inside
/// `vision_config`, so they are parsed and retained here for the fusion stage
/// to read back off a single struct.
#[derive(Debug, Clone)]
pub struct Florence2VisionConfig {
    pub depths: Vec<i32>,
    pub dim_embed: Vec<i32>,
    pub num_heads: Vec<i32>,
    pub num_groups: Vec<i32>,
    pub patch_size: Vec<i32>,
    pub patch_stride: Vec<i32>,
    pub patch_padding: Vec<i32>,
    pub patch_prenorm: Vec<bool>,
    pub window_size: i32,
    pub in_chans: i32,
    pub mlp_ratio: f32,
    pub qkv_bias: bool,
    pub conv_at_attn: bool,
    pub conv_at_ffn: bool,
    /// Stochastic-depth rate. Training-only: `DropPath` is the identity at
    /// inference, so this never affects the forward here. Parsed so the
    /// config round-trips faithfully.
    pub drop_path_rate: f32,
    /// Fusion-stage width the backbone output is projected down to
    /// (1024 -> 768 for `Florence-2-base-ft`). Not applied here.
    pub projection_dim: i32,
    /// Fusion-stage learned 2-D position embedding spec, e.g.
    /// `{"type": "learned_abs_2d", "max_pos_embeddings": 50}`.
    pub image_pos_embed: Option<Value>,
    /// Fusion-stage temporal embedding spec, e.g.
    /// `{"type": "COSINE", "max_temporal_embeddings": 100}`.
    pub visual_temporal_embedding: Option<Value>,
    /// Fusion-stage pooling recipe, e.g.
    /// `["spatial_avg_pool", "temporal_avg_pool"]`.
    pub image_feature_source: Vec<String>,
    /// Packing of the tower's attention and MLP projections.
    /// [`Florence2Quantization::DENSE`] for a bf16 or f16 export.
    ///
    /// The block lives at the top level of a Florence-2 `config.json`, not
    /// inside `vision_config`, so only [`Self::from_model_config`] can fill
    /// it; [`Self::from_vision_config`] leaves it at the dense default
    /// because the sub-object it is handed cannot carry it.
    pub quantization: Florence2Quantization,
}

fn i32_list(config: &Value, key: &str) -> Option<Vec<i32>> {
    config.get(key)?.as_array().map(|items| {
        items
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect()
    })
}

fn bool_list(config: &Value, key: &str) -> Option<Vec<bool>> {
    config
        .get(key)?
        .as_array()
        .map(|items| items.iter().filter_map(Value::as_bool).collect())
}

impl Florence2VisionConfig {
    /// Parse from the `vision_config` sub-object itself.
    ///
    /// Real Florence-2 checkpoints ship `vision_config.model_type` as the
    /// empty string, so gating on the literal `"davit"` would reject every
    /// one of them. Accept `""`, `"davit"`, and an absent key (mirroring the
    /// upstream `VisionModel.__init__` guard, which allows `["davit", ""]`)
    /// and reject anything else. The DaViT family is identified by the parent
    /// `model_type: florence2` plus the structural fields required below.
    pub fn from_vision_config(config: &Value) -> Result<Self> {
        if let Some(model_type) = config.get("model_type").and_then(Value::as_str)
            && !model_type.is_empty()
            && model_type != "davit"
        {
            return Err(anyhow!(
                "Florence-2 vision_config model_type {model_type:?} is not a DaViT backbone"
            ));
        }

        let require_i32s = |key: &str| -> Result<Vec<i32>> {
            i32_list(config, key)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| anyhow!("Florence-2 vision_config missing field: {key}"))
        };

        let depths = require_i32s("depths")?;
        let dim_embed = require_i32s("dim_embed")?;
        let num_heads = require_i32s("num_heads")?;
        let num_groups = i32_list(config, "num_groups")
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| num_heads.clone());
        let stages = dim_embed.len();

        let parsed = Self {
            depths,
            dim_embed,
            num_heads,
            num_groups,
            patch_size: require_i32s("patch_size")?,
            patch_stride: require_i32s("patch_stride")?,
            patch_padding: require_i32s("patch_padding")?,
            patch_prenorm: bool_list(config, "patch_prenorm")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| vec![false; stages]),
            window_size: config
                .get("window_size")
                .and_then(Value::as_i64)
                .unwrap_or(12) as i32,
            in_chans: config.get("in_chans").and_then(Value::as_i64).unwrap_or(3) as i32,
            mlp_ratio: config
                .get("mlp_ratio")
                .and_then(Value::as_f64)
                .unwrap_or(4.0) as f32,
            qkv_bias: config
                .get("qkv_bias")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            conv_at_attn: config
                .get("conv_at_attn")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            conv_at_ffn: config
                .get("conv_at_ffn")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            drop_path_rate: config
                .get("drop_path_rate")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32,
            projection_dim: config
                .get("projection_dim")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32,
            image_pos_embed: config.get("image_pos_embed").cloned(),
            visual_temporal_embedding: config.get("visual_temporal_embedding").cloned(),
            image_feature_source: config
                .get("image_feature_source")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            quantization: Florence2Quantization::DENSE,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<()> {
        let stages = self.dim_embed.len();
        let mismatched = [
            ("depths", self.depths.len()),
            ("num_heads", self.num_heads.len()),
            ("num_groups", self.num_groups.len()),
            ("patch_size", self.patch_size.len()),
            ("patch_stride", self.patch_stride.len()),
            ("patch_padding", self.patch_padding.len()),
            ("patch_prenorm", self.patch_prenorm.len()),
        ]
        .into_iter()
        .find(|(_, len)| *len != stages);
        if let Some((field, len)) = mismatched {
            return Err(anyhow!(
                "Florence-2 vision_config {field} has {len} entries but dim_embed has {stages}"
            ));
        }
        for (stage, (&dim, (&heads, &groups))) in self
            .dim_embed
            .iter()
            .zip(self.num_heads.iter().zip(self.num_groups.iter()))
            .enumerate()
        {
            if heads <= 0 || groups <= 0 || dim % heads != 0 || dim % groups != 0 {
                return Err(anyhow!(
                    "Florence-2 vision_config stage {stage}: dim_embed {dim} not divisible by num_heads {heads} / num_groups {groups}"
                ));
            }
        }
        for (stage, (&depth, ((&patch, &stride), &padding))) in self
            .depths
            .iter()
            .zip(
                self.patch_size
                    .iter()
                    .zip(self.patch_stride.iter())
                    .zip(self.patch_padding.iter()),
            )
            .enumerate()
        {
            if !(0..=MAX_STAGE_DEPTH).contains(&depth) {
                return Err(anyhow!(
                    "Florence-2 vision_config stage {stage}: depths {depth} outside 0..={MAX_STAGE_DEPTH}"
                ));
            }
            // These three reach MLX `conv2d` directly. MLX validates them
            // eagerly and throws, and the throw crosses an FFI boundary that
            // cannot carry it, so they have to be rejected here.
            if patch < 1 || stride < 1 || padding < 0 {
                return Err(anyhow!(
                    "Florence-2 vision_config stage {stage}: patch_size {patch} must be >= 1, patch_stride {stride} must be >= 1, patch_padding {padding} must be >= 0"
                ));
            }
        }
        if self.window_size <= 0 {
            return Err(anyhow!(
                "Florence-2 vision_config window_size must be positive, got {}",
                self.window_size
            ));
        }
        if self.in_chans < 1 {
            return Err(anyhow!(
                "Florence-2 vision_config in_chans must be positive, got {}",
                self.in_chans
            ));
        }
        Ok(())
    }

    /// Parse from a full Florence-2 `config.json` (`model_type: florence2`),
    /// descending into its `vision_config` sub-object. A bare vision config
    /// (no `vision_config` key) is also accepted so the sub-object can be
    /// passed directly, mirroring
    /// [`crate::models::Florence2TextConfig::from_model_config`].
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let mut parsed = match config.get("vision_config") {
            Some(vision) => Self::from_vision_config(vision)?,
            None => Self::from_vision_config(config)?,
        };
        // The `quantization` block is a property of the whole checkpoint and
        // sits beside `vision_config`, not inside it, so it can only be read
        // from the full document.
        parsed.quantization = Florence2Quantization::from_model_config(config)?;
        Ok(parsed)
    }

    /// Number of DaViT stages.
    pub fn num_stages(&self) -> usize {
        self.dim_embed.len()
    }

    /// Channel width of the backbone output (`dim_embed[-1]`, 1024 for
    /// `Florence-2-base-ft`). This is *not* `projection_dim`; the fusion
    /// stage projects this down.
    pub fn output_dim(&self) -> i32 {
        self.dim_embed.last().copied().unwrap_or(0)
    }
}

/// True when a 4-D conv weight is already in MLX channels-last layout.
///
/// Heuristic ported verbatim from the reference `check_array_shape`: a
/// `(out, kH, kW, in)` tensor has `out` at least as large as both spatial
/// extents and a square kernel, which no `(out, in, kH, kW)` PyTorch tensor
/// of this family satisfies (its axis 1 is the input channel count, which
/// differs from the kernel width). Makes [`sanitize`] idempotent.
fn is_channels_last_conv(shape: &[i32]) -> bool {
    if shape.len() != 4 {
        return false;
    }
    let (out_channels, kh, kw) = (shape[0], shape[1], shape[2]);
    out_channels >= kh && out_channels >= kw && kh == kw
}

/// Remap DaViT conv weights to MLX channels-last layout and drop unused
/// `position_ids` buffers.
///
/// PyTorch stores conv weights as `(out, in / groups, kH, kW)` while MLX
/// wants `(out, kH, kW, in / groups)`. Both affected key families are guarded
/// so the pass is a no-op on checkpoints that already ship the MLX layout
/// (the `mlx-community` bf16 exports do).
pub fn sanitize(weights: WeightMap) -> WeightMap {
    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        if key.contains("position_ids") {
            continue;
        }
        let shape = mlxcel_core::array_shape(&value);
        let remapped = if key.contains("convs") && key.ends_with("proj.weight") {
            if is_channels_last_conv(&shape) {
                value
            } else {
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            }
        } else if key.contains("blocks") && key.ends_with("dw.weight") {
            // Depthwise: `(C, 1, kH, kW)` -> `(C, kH, kW, 1)`. Axis 1 is the
            // per-group input count (1), so it is smaller than the kernel
            // width exactly when the tensor is still PyTorch-ordered.
            if shape.len() == 4 && shape[1] < shape[3] {
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            } else {
                value
            }
        } else {
            value
        };
        out.insert(key, remapped);
    }
    out
}

/// Loaded DaViT backbone: one `ConvEmbed` and `depths[i]` blocks per stage.
pub struct Florence2DaViT {
    config: Florence2VisionConfig,
    convs: Vec<ConvEmbed>,
    blocks: Vec<Vec<Block>>,
}

impl Florence2DaViT {
    /// Build the backbone from an already-loaded [`WeightMap`].
    ///
    /// `prefix` is the key prefix in front of `convs.` / `blocks.`
    /// ([`FLORENCE2_VISION_PREFIX`] inside a full Florence-2 checkpoint, `""`
    /// for a bare tower export). The weights must already have been passed
    /// through [`sanitize`]; every conv weight is checked against the layout
    /// that pass produces, so a raw PyTorch export fails here with the
    /// offending key named instead of reaching MLX and aborting the process.
    pub fn from_weights(
        weights: &WeightMap,
        config: &Florence2VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        // `Florence2VisionConfig` has public fields, so a caller can hand this
        // a struct that never went through `from_vision_config`. Re-check
        // before any of it is used as a divisor or an allocation size.
        config.validate().map_err(|e| e.to_string())?;

        let mut convs = Vec::with_capacity(config.num_stages());
        let mut blocks = Vec::with_capacity(config.num_stages());

        for stage in 0..config.num_stages() {
            let in_channels = if stage == 0 {
                config.in_chans
            } else {
                config.dim_embed[stage - 1]
            };
            convs.push(ConvEmbed::from_weights(
                weights,
                &format!("{prefix}convs.{stage}"),
                in_channels,
                config.dim_embed[stage],
                config.patch_stride[stage],
                config.patch_padding[stage],
                config.patch_prenorm[stage],
            )?);

            let params = BlockParams {
                dim: config.dim_embed[stage],
                num_heads: config.num_heads[stage],
                num_groups: config.num_groups[stage],
                window_size: config.window_size,
                conv_at_attn: config.conv_at_attn,
                conv_at_ffn: config.conv_at_ffn,
                quantization: config.quantization,
            };
            let mut stage_blocks = Vec::with_capacity(config.depths[stage].max(0) as usize);
            for depth in 0..config.depths[stage] {
                stage_blocks.push(Block::from_weights(
                    weights,
                    &format!("{prefix}blocks.{stage}.{depth}"),
                    params,
                )?);
            }
            blocks.push(stage_blocks);
        }

        Ok(Self {
            config: config.clone(),
            convs,
            blocks,
        })
    }

    /// Load the backbone from a Florence-2 checkpoint directory
    /// (`config.json` + safetensors). Only `vision_tower.*` tensors are read;
    /// the language model and the fusion-stage projection / position
    /// embeddings are left on disk for their owners.
    pub fn load(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read {config_path:?}: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let config: Value = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse Florence-2 config: {e}"))?;
        let vision_config = Florence2VisionConfig::from_model_config(&config)?;

        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_path, |k| {
            k.starts_with(FLORENCE2_VISION_PREFIX)
        })
        .map_err(|e| anyhow!("Failed to load Florence-2 vision weights: {e}"))?;
        let mut weights = sanitize(weights);
        // The tower's own dense-only tensors are the conv stack; a checkpoint
        // that packed one would reach `conv2d` as `uint32` and abort.
        crate::models::florence2::reject_unsupported_quantized_tensors(&weights)
            .map_err(|e| anyhow!("{e}"))?;
        // Apple Silicon precision policy: bf16 -> f16, but only for a dense
        // export. A quantized one keeps its scales and biases at the width the
        // checkpoint stored them, because those are dequantization operands
        // rather than activations and rounding them changes every weight the
        // tower reconstructs.
        if !Florence2Quantization::config_is_quantized(&config) {
            let _ = crate::models::convert_bf16_weights(&mut weights);
        }

        Self::from_weights(&weights, &vision_config, FLORENCE2_VISION_PREFIX)
            .map_err(|e| anyhow!("Failed to build Florence-2 DaViT backbone: {e}"))
    }

    /// Parsed vision configuration.
    pub fn config(&self) -> &Florence2VisionConfig {
        &self.config
    }

    /// Run every stage, returning the token tensor and `(H, W)` grid after
    /// each one. The last entry is what [`Self::forward`] returns.
    ///
    /// `pixel_values` is NCHW `(B, in_chans, H, W)`, matching the reference,
    /// which reads its input size from `x.shape[2:]`.
    pub fn forward_stages(
        &self,
        pixel_values: &MlxArray,
    ) -> Vec<(UniquePtr<MlxArray>, (i32, i32))> {
        let shape = mlxcel_core::array_shape(pixel_values);
        let mut size = (shape[2], shape[3]);
        let mut current: Option<UniquePtr<MlxArray>> = None;
        let mut stages = Vec::with_capacity(self.convs.len());

        for (conv, stage_blocks) in self.convs.iter().zip(self.blocks.iter()) {
            let (embedded, new_size) =
                conv.forward(current.as_deref().unwrap_or(pixel_values), size);
            size = new_size;
            let mut stage_out = embedded;
            for block in stage_blocks {
                stage_out = block.forward(&stage_out, size);
            }
            stages.push((mlxcel_core::copy(&stage_out), size));
            current = Some(stage_out);
        }
        stages
    }

    /// Encode `pixel_values` NCHW `(B, in_chans, H, W)` into image features
    /// `(B, N, dim_embed[-1])`.
    pub fn forward(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(pixel_values);
        let mut size = (shape[2], shape[3]);
        let mut current: Option<UniquePtr<MlxArray>> = None;
        for (conv, stage_blocks) in self.convs.iter().zip(self.blocks.iter()) {
            let (embedded, new_size) =
                conv.forward(current.as_deref().unwrap_or(pixel_values), size);
            size = new_size;
            let mut stage_out = embedded;
            for block in stage_blocks {
                stage_out = block.forward(&stage_out, size);
            }
            current = Some(stage_out);
        }
        // Validation rejects an empty `dim_embed`, so the fallback is only
        // reachable for a hand-built zero-stage backbone.
        current.unwrap_or_else(|| mlxcel_core::copy(pixel_values))
    }
}

impl VisionEncoder for Florence2DaViT {
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        VisionEncoderOutput {
            hidden_states: Florence2DaViT::forward(self, pixel_values),
        }
    }
}

#[cfg(test)]
#[path = "florence2_davit_tests.rs"]
mod florence2_davit_tests;
