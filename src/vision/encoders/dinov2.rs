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

//! Shared DINOv2 image encoder for standalone features and dense models.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::dense::interpolate::{resize_bicubic_nhwc, resize_bilinear_nhwc};
use crate::vision::dense::preprocess::normalize_imagenet;

pub use super::dinov2_config::Dinov2Config;
pub use super::dinov2_layers::{Attention, Mlp};
use super::dinov2_layers::{Block, PatchEmbed, check_shape, load_norm, take_weight};
use super::dinov2_weights::sanitize_weights;

pub struct Dinov2Features {
    pub x_norm: UniquePtr<MlxArray>,
    pub x_norm_cls_token: UniquePtr<MlxArray>,
    pub x_norm_register_tokens: UniquePtr<MlxArray>,
    pub x_norm_patch_tokens: UniquePtr<MlxArray>,
    pub x_prenorm: UniquePtr<MlxArray>,
}

pub struct Dinov2Output {
    pub last_hidden_state: UniquePtr<MlxArray>,
    pub pooler_output: UniquePtr<MlxArray>,
    pub hidden_patch_tokens: UniquePtr<MlxArray>,
}

// Used by: Dinov2, VideoDepthAnything, MoGe3, Sam3dObjects
pub struct Dinov2 {
    pub config: Dinov2Config,
    patch_embed: PatchEmbed,
    cls_token: UniquePtr<MlxArray>,
    pos_embed: UniquePtr<MlxArray>,
    mask_token: UniquePtr<MlxArray>,
    register_tokens: Option<UniquePtr<MlxArray>>,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

impl Dinov2 {
    pub fn load(path: &Path) -> Result<Self> {
        let config = Dinov2Config::from_path(path)?;
        let weights = mlxcel_core::weights::load_weights_from_dir(path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("load DINOv2 weights from {}", path.display()))?;
        let mut weights = sanitize_weights(weights).context("sanitize DINOv2 weights")?;
        let _ = crate::models::convert_bf16_weights(&mut weights);
        Self::from_weights(&weights, config).map_err(anyhow::Error::msg)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: Dinov2Config,
    ) -> std::result::Result<Self, String> {
        config.validate().map_err(|error| error.to_string())?;
        let hidden = config.hidden_size;
        let base_grid = config.image_size / config.patch_size;
        let cls_token = take_weight(weights, "cls_token")?;
        let pos_embed = take_weight(weights, "pos_embed")?;
        let mask_token = take_weight(weights, "mask_token")?;
        check_shape(&cls_token, &[1, 1, hidden as i32], "cls_token")?;
        check_shape(
            &pos_embed,
            &[1, (base_grid * base_grid + 1) as i32, hidden as i32],
            "pos_embed",
        )?;
        let mask_shape = mlxcel_core::array_shape(&mask_token);
        if mask_shape != [1, hidden as i32] && mask_shape != [1, 1, hidden as i32] {
            return Err(format!(
                "DINOv2 mask_token: expected [1, {hidden}] or [1, 1, {hidden}], got {mask_shape:?}"
            ));
        }
        let register_tokens = if config.num_register_tokens == 0 {
            None
        } else {
            let value = take_weight(weights, "register_tokens")?;
            check_shape(
                &value,
                &[1, config.num_register_tokens as i32, hidden as i32],
                "register_tokens",
            )?;
            Some(value)
        };
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            blocks.push(Block::from_weights(weights, &config, index)?);
        }
        Ok(Self {
            patch_embed: PatchEmbed::from_weights(weights, &config)?,
            norm: load_norm(weights, "norm", hidden, config.layer_norm_eps)?,
            config,
            cls_token,
            pos_embed,
            mask_token,
            register_tokens,
            blocks,
        })
    }

    pub fn interpolate_pos_encoding(
        &self,
        tokens: &MlxArray,
        height: usize,
        width: usize,
    ) -> Result<UniquePtr<MlxArray>> {
        let token_shape = mlxcel_core::array_shape(tokens);
        ensure!(
            token_shape.len() == 3
                && token_shape[1] > 0
                && token_shape[2] == self.config.hidden_size as i32,
            "DINOv2 tokens must have shape [batch, tokens, {}] with at least one token",
            self.config.hidden_size
        );
        ensure!(
            height > 0
                && width > 0
                && height.is_multiple_of(self.config.patch_size)
                && width.is_multiple_of(self.config.patch_size),
            "DINOv2 interpolation size must be a positive patch multiple"
        );
        let target_patches = (height / self.config.patch_size)
            .checked_mul(width / self.config.patch_size)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 target patch count overflowed"))?;
        ensure!(
            target_patches <= i32::MAX as usize && token_shape[1] as usize == target_patches + 1,
            "DINOv2 token count does not match the requested image grid"
        );
        let positions = mlxcel_core::array_shape(&self.pos_embed)[1] as usize - 1;
        if token_shape[1] as usize - 1 == positions && height == width {
            return Ok(mlxcel_core::copy(&self.pos_embed));
        }
        let side = (positions as f64).sqrt() as usize;
        ensure!(
            side * side == positions,
            "DINOv2 patch position count {positions} is not square"
        );
        let hidden = self.config.hidden_size as i32;
        let class = mlxcel_core::slice(&self.pos_embed, &[0, 0, 0], &[1, 1, hidden]);
        let patches = mlxcel_core::slice(
            &self.pos_embed,
            &[0, 1, 0],
            &[1, positions as i32 + 1, hidden],
        );
        let patches = mlxcel_core::reshape(&patches, &[1, side as i32, side as i32, hidden]);
        let patches = mlxcel_core::astype(&patches, mlxcel_core::dtype::FLOAT32);
        let patches = if self.config.interpolate_offset > 0.0 {
            let offset = self.config.interpolate_offset;
            resize_bicubic_nhwc(
                &patches,
                None,
                Some((
                    ((height / self.config.patch_size) as f32 + offset) / side as f32,
                    ((width / self.config.patch_size) as f32 + offset) / side as f32,
                )),
                false,
            )?
        } else {
            resize_bicubic_nhwc(
                &patches,
                Some((
                    height / self.config.patch_size,
                    width / self.config.patch_size,
                )),
                None,
                self.config.interpolate_antialias,
            )?
        };
        let patches = mlxcel_core::reshape(&patches, &[1, -1, hidden]);
        let positions = mlxcel_core::concatenate(&class, &patches, 1);
        Ok(mlxcel_core::astype(
            &positions,
            mlxcel_core::array_dtype(tokens),
        ))
    }

    pub fn prepare_tokens(
        &self,
        image: &MlxArray,
        bool_masked_pos: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let image_shape = mlxcel_core::array_shape(image);
        let (batch, height, width) = (
            image_shape.first().copied().unwrap_or(0),
            image_shape.get(1).copied().unwrap_or(0),
            image_shape.get(2).copied().unwrap_or(0),
        );
        let mut patches = self.patch_embed.forward(image)?;
        let patch_shape = mlxcel_core::array_shape(&patches);
        let runtime_tokens = (patch_shape[1] as usize)
            .checked_add(self.config.num_register_tokens)
            .and_then(|tokens| tokens.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("DINOv2 runtime token count overflowed"))?;
        ensure!(
            runtime_tokens <= i32::MAX as usize,
            "DINOv2 runtime token count exceeds the MLX shape range"
        );
        if let Some(mask) = bool_masked_pos {
            ensure!(
                mlxcel_core::array_shape(mask) == [batch, patch_shape[1]],
                "DINOv2 bool_masked_pos must have shape [{batch}, {}]",
                patch_shape[1]
            );
            ensure!(
                mlxcel_core::array_dtype(mask) == mlxcel_core::dtype::BOOL,
                "DINOv2 bool_masked_pos must have bool dtype"
            );
            let mask = mlxcel_core::reshape(mask, &[batch, patch_shape[1], 1]);
            let token =
                mlxcel_core::reshape(&self.mask_token, &[1, 1, self.config.hidden_size as i32]);
            let token = mlxcel_core::broadcast_to(
                &token,
                &[batch, patch_shape[1], self.config.hidden_size as i32],
            );
            patches = mlxcel_core::where_cond(&mask, &token, &patches);
        }
        let cls =
            mlxcel_core::broadcast_to(&self.cls_token, &[batch, 1, self.config.hidden_size as i32]);
        let mut tokens = mlxcel_core::concatenate(&cls, &patches, 1);
        let positions = self.interpolate_pos_encoding(&tokens, height as usize, width as usize)?;
        tokens = mlxcel_core::add(&tokens, &positions);
        if let Some(registers) = &self.register_tokens {
            let registers = mlxcel_core::broadcast_to(
                registers,
                &[
                    batch,
                    self.config.num_register_tokens as i32,
                    self.config.hidden_size as i32,
                ],
            );
            let class = mlxcel_core::slice(
                &tokens,
                &[0, 0, 0],
                &[batch, 1, self.config.hidden_size as i32],
            );
            let patches = mlxcel_core::slice(
                &tokens,
                &[0, 1, 0],
                &[batch, patch_shape[1] + 1, self.config.hidden_size as i32],
            );
            tokens = mlxcel_core::concatenate_many(&[&class, &registers, &patches], 1);
        }
        Ok(tokens)
    }

    fn encode(
        &self,
        image: &MlxArray,
        bool_masked_pos: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let mut hidden = self.prepare_tokens(image, bool_masked_pos)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        Ok(hidden)
    }

    pub fn forward_features(
        &self,
        image: &MlxArray,
        bool_masked_pos: Option<&MlxArray>,
    ) -> Result<Dinov2Features> {
        let prenorm = self.encode(image, bool_masked_pos)?;
        let normed = self.norm.forward(&prenorm);
        let shape = mlxcel_core::array_shape(&normed);
        let registers = self.config.num_register_tokens as i32;
        let cls = mlxcel_core::slice(&normed, &[0, 0, 0], &[shape[0], 1, shape[2]]);
        let cls = mlxcel_core::reshape(&cls, &[shape[0], shape[2]]);
        let register_tokens = if registers == 0 {
            mlxcel_core::zeros(&[shape[0], 0, shape[2]], mlxcel_core::array_dtype(&normed))
        } else {
            mlxcel_core::slice(&normed, &[0, 1, 0], &[shape[0], 1 + registers, shape[2]])
        };
        let patch_tokens = mlxcel_core::slice(
            &normed,
            &[0, 1 + registers, 0],
            &[shape[0], shape[1], shape[2]],
        );
        Ok(Dinov2Features {
            x_norm: normed,
            x_norm_cls_token: cls,
            x_norm_register_tokens: register_tokens,
            x_norm_patch_tokens: patch_tokens,
            x_prenorm: prenorm,
        })
    }

    pub fn forward(
        &self,
        image: &MlxArray,
        bool_masked_pos: Option<&MlxArray>,
    ) -> Result<Dinov2Output> {
        let features = self.forward_features(image, bool_masked_pos)?;
        Ok(Dinov2Output {
            last_hidden_state: features.x_norm,
            pooler_output: features.x_norm_cls_token,
            hidden_patch_tokens: features.x_norm_patch_tokens,
        })
    }

    pub fn get_intermediate_layers(
        &self,
        image: &MlxArray,
        indices: &[usize],
    ) -> Result<Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>> {
        for &index in indices {
            ensure!(
                index < self.blocks.len(),
                "DINOv2 intermediate layer index {index} is outside depth {}",
                self.blocks.len()
            );
        }
        let mut outputs: Vec<Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>> =
            (0..indices.len()).map(|_| None).collect();
        let mut hidden = self.prepare_tokens(image, None)?;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden);
            let requested: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter_map(|(position, requested)| (*requested == index).then_some(position))
                .collect();
            if requested.is_empty() {
                continue;
            }
            let normalized = self.norm.forward(&hidden);
            let shape = mlxcel_core::array_shape(&normalized);
            for position in requested {
                let patches = mlxcel_core::slice(
                    &normalized,
                    &[0, 1 + self.config.num_register_tokens as i32, 0],
                    &[shape[0], shape[1], shape[2]],
                );
                let cls = mlxcel_core::slice(&normalized, &[0, 0, 0], &[shape[0], 1, shape[2]]);
                outputs[position] =
                    Some((patches, mlxcel_core::reshape(&cls, &[shape[0], shape[2]])));
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| anyhow::anyhow!("DINOv2 intermediate output missing"))
            })
            .collect()
    }
}

// Used by: Dinov2, VideoDepthAnything, MoGe3, Sam3dObjects
pub struct Dinov2Encoder {
    pub config: Dinov2Config,
    pub backbone: Dinov2,
}

/// Upstream-compatible spelling retained for dense-model consumers.
pub type DINOv2Encoder = Dinov2Encoder;

impl Dinov2Encoder {
    pub fn new(backbone: Dinov2) -> Self {
        Self {
            config: backbone.config.clone(),
            backbone,
        }
    }

    pub fn call(
        &self,
        image_01_nhwc: &MlxArray,
        token_rows: usize,
        token_cols: usize,
    ) -> Result<Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>> {
        ensure!(
            token_rows > 0 && token_cols > 0,
            "DINOv2 token grid must be positive"
        );
        let resized_rows = token_rows
            .checked_mul(self.config.patch_size)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 token row size overflowed"))?;
        let resized_cols = token_cols
            .checked_mul(self.config.patch_size)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 token column size overflowed"))?;
        ensure!(
            resized_rows <= i32::MAX as usize && resized_cols <= i32::MAX as usize,
            "DINOv2 token grid exceeds the MLX shape range"
        );
        let shape = mlxcel_core::array_shape(image_01_nhwc);
        ensure!(
            shape.len() == 4 && shape[3] == 3,
            "DINOv2Encoder expects NHWC RGB"
        );
        let resized =
            resize_bilinear_nhwc(image_01_nhwc, (resized_rows, resized_cols), false, true)?;
        let normalized = normalize_imagenet(&resized)?;
        let batch = shape[0];
        self.backbone
            .get_intermediate_layers(&normalized, &self.config.intermediate_layers)?
            .into_iter()
            .map(|(tokens, cls)| {
                Ok((
                    mlxcel_core::reshape(
                        &tokens,
                        &[
                            batch,
                            token_rows as i32,
                            token_cols as i32,
                            self.config.hidden_size as i32,
                        ],
                    ),
                    cls,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "dinov2_tests.rs"]
mod tests;
