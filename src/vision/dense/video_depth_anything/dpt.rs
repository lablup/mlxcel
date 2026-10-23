// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Temporal DPT decoder for Video Depth Anything.

use anyhow::{Result, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::ResolvedVdaConfig;
use super::motion::TemporalModule;
use super::weights::{self, Conv2d, ConvTranspose2d};
use crate::vision::dense::interpolate::resize_bilinear_nhwc;

fn upsample(x: &MlxArray, size: (usize, usize)) -> Result<UniquePtr<MlxArray>> {
    resize_bilinear_nhwc(x, size, true, false)
}

struct ResidualConvUnit {
    first: Conv2d,
    second: Conv2d,
}

impl ResidualConvUnit {
    fn from_weights(map: &WeightMap, prefix: &str, features: usize) -> Result<Self> {
        Ok(Self {
            first: weights::conv(map, &format!("{prefix}.conv1"), features, features, 3, true)?,
            second: weights::conv(map, &format!("{prefix}.conv2"), features, features, 3, true)?,
        })
    }
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let first = self.first.forward(&mlxcel_core::relu(x), 1, 1);
        mlxcel_core::add(x, &self.second.forward(&mlxcel_core::relu(&first), 1, 1))
    }
}

struct FeatureFusionBlock {
    output: Conv2d,
    first: ResidualConvUnit,
    second: ResidualConvUnit,
}

impl FeatureFusionBlock {
    fn from_weights(map: &WeightMap, prefix: &str, features: usize) -> Result<Self> {
        Ok(Self {
            output: weights::conv(
                map,
                &format!("{prefix}.out_conv"),
                features,
                features,
                1,
                true,
            )?,
            first: ResidualConvUnit::from_weights(
                map,
                &format!("{prefix}.resConfUnit1"),
                features,
            )?,
            second: ResidualConvUnit::from_weights(
                map,
                &format!("{prefix}.resConfUnit2"),
                features,
            )?,
        })
    }
    fn forward(
        &self,
        x: &MlxArray,
        residual: Option<&MlxArray>,
        size: Option<(usize, usize)>,
    ) -> Result<UniquePtr<MlxArray>> {
        let mut output = mlxcel_core::copy(x);
        if let Some(residual) = residual {
            output = mlxcel_core::add(&output, &self.first.forward(residual));
        }
        output = self.second.forward(&output);
        let shape = mlxcel_core::array_shape(&output);
        output = upsample(
            &output,
            size.unwrap_or((shape[1] as usize * 2, shape[2] as usize * 2)),
        )?;
        Ok(self.output.forward(&output, 1, 0))
    }
}

enum ResizeLayer {
    Transposed(ConvTranspose2d, i32),
    Identity,
    Downsample(Conv2d),
}

impl ResizeLayer {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Transposed(layer, stride) => layer.forward(x, *stride),
            Self::Identity => mlxcel_core::copy(x),
            Self::Downsample(layer) => layer.forward(x, 2, 1),
        }
    }
}

pub struct DptHead {
    patch_size: usize,
    projects: Vec<Conv2d>,
    resize: Vec<ResizeLayer>,
    layer_rn: Vec<Conv2d>,
    refine: Vec<FeatureFusionBlock>,
    output1: Conv2d,
    output2_first: Conv2d,
    output2_last: Conv2d,
    motion: Vec<TemporalModule>,
}

impl DptHead {
    pub fn from_weights(map: &WeightMap, config: &ResolvedVdaConfig) -> Result<Self> {
        let mut projects = Vec::with_capacity(4);
        let mut layer_rn = Vec::with_capacity(4);
        for index in 0..4 {
            projects.push(weights::conv(
                map,
                &format!("head.projects.{index}"),
                config.dinov2.hidden_size,
                config.out_channels[index],
                1,
                true,
            )?);
            layer_rn.push(weights::conv(
                map,
                &format!("head.scratch.layer{}_rn", index + 1),
                config.out_channels[index],
                config.features,
                3,
                false,
            )?);
        }
        let resize = vec![
            ResizeLayer::Transposed(
                ConvTranspose2d::from_weights(
                    map,
                    "head.resize_layers.0",
                    config.out_channels[0],
                    4,
                )?,
                4,
            ),
            ResizeLayer::Transposed(
                ConvTranspose2d::from_weights(
                    map,
                    "head.resize_layers.1",
                    config.out_channels[1],
                    2,
                )?,
                2,
            ),
            ResizeLayer::Identity,
            ResizeLayer::Downsample(weights::conv(
                map,
                "head.resize_layers.3",
                config.out_channels[3],
                config.out_channels[3],
                3,
                true,
            )?),
        ];
        let mut refine = Vec::with_capacity(4);
        for index in 1..=4 {
            refine.push(FeatureFusionBlock::from_weights(
                map,
                &format!("head.scratch.refinenet{index}"),
                config.features,
            )?);
        }
        let motion_channels = [
            config.out_channels[2],
            config.out_channels[3],
            config.features,
            config.features,
        ];
        let mut motion = Vec::with_capacity(4);
        for (index, channels) in motion_channels.into_iter().enumerate() {
            motion.push(TemporalModule::from_weights(
                map,
                &format!("head.motion_modules.{index}"),
                channels,
                config.num_attention_heads,
                config.norm_num_groups,
                config.num_transformer_block,
                config.num_attention_blocks,
            )?);
        }
        Ok(Self {
            patch_size: config.dinov2.patch_size,
            projects,
            resize,
            layer_rn,
            refine,
            output1: weights::conv(
                map,
                "head.scratch.output_conv1",
                config.features,
                config.features / 2,
                3,
                true,
            )?,
            output2_first: weights::conv(
                map,
                "head.scratch.output_conv2.0",
                config.features / 2,
                32,
                3,
                true,
            )?,
            output2_last: weights::conv(map, "head.scratch.output_conv2.2", 32, 1, 1, true)?,
            motion,
        })
    }

    fn apply_motion(&self, index: usize, x: &MlxArray, frames: usize) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0] / frames as i32;
        let grouped =
            mlxcel_core::reshape(x, &[batch, frames as i32, shape[1], shape[2], shape[3]]);
        let output = self.motion[index].forward(&grouped, frames);
        mlxcel_core::reshape(&output, &shape)
    }

    pub fn forward(
        &self,
        features: &[(UniquePtr<MlxArray>, UniquePtr<MlxArray>)],
        patch_h: usize,
        patch_w: usize,
        frames: usize,
    ) -> Result<UniquePtr<MlxArray>> {
        ensure!(
            features.len() == 4,
            "VDA DPT requires four backbone feature levels"
        );
        let mut levels = Vec::with_capacity(4);
        for (index, (tokens, _)) in features.iter().enumerate() {
            let shape = mlxcel_core::array_shape(tokens);
            ensure!(
                shape.len() == 3 && shape[1] as usize == patch_h * patch_w,
                "VDA feature {index} has unexpected shape {shape:?}"
            );
            let grid = mlxcel_core::reshape(
                tokens,
                &[shape[0], patch_h as i32, patch_w as i32, shape[2]],
            );
            levels.push(self.resize[index].forward(&self.projects[index].forward(&grid, 1, 0)));
        }
        levels[2] = self.apply_motion(0, &levels[2], frames);
        levels[3] = self.apply_motion(1, &levels[3], frames);
        let rn: Vec<UniquePtr<MlxArray>> = levels
            .iter()
            .zip(&self.layer_rn)
            .map(|(level, conv)| conv.forward(level, 1, 1))
            .collect();
        let size3 = mlxcel_core::array_shape(&rn[2]);
        let mut path4 =
            self.refine[3].forward(&rn[3], None, Some((size3[1] as usize, size3[2] as usize)))?;
        path4 = self.apply_motion(2, &path4, frames);
        let size2 = mlxcel_core::array_shape(&rn[1]);
        let mut path3 = self.refine[2].forward(
            &path4,
            Some(&rn[2]),
            Some((size2[1] as usize, size2[2] as usize)),
        )?;
        path3 = self.apply_motion(3, &path3, frames);
        self.output_head(&path3, &rn[1], &rn[0], patch_h, patch_w)
    }

    fn output_head(
        &self,
        path3: &MlxArray,
        layer2: &MlxArray,
        layer1: &MlxArray,
        patch_h: usize,
        patch_w: usize,
    ) -> Result<UniquePtr<MlxArray>> {
        let count = mlxcel_core::array_shape(path3)[0];
        ensure!(
            mlxcel_core::array_shape(layer2)[0] == count
                && mlxcel_core::array_shape(layer1)[0] == count,
            "VDA high-resolution head batches do not match"
        );
        let mut chunks = Vec::new();
        for start in (0..count).step_by(4) {
            let end = (start + 4).min(count);
            let slice_batch = |value: &MlxArray| {
                let shape = mlxcel_core::array_shape(value);
                mlxcel_core::slice(
                    value,
                    &[start, 0, 0, 0],
                    &[end, shape[1], shape[2], shape[3]],
                )
            };
            let path3 = slice_batch(path3);
            let layer2 = slice_batch(layer2);
            let layer1 = slice_batch(layer1);
            let size1 = mlxcel_core::array_shape(&layer1);
            let path2 = self.refine[1].forward(
                &path3,
                Some(&layer2),
                Some((size1[1] as usize, size1[2] as usize)),
            )?;
            let path1 = self.refine[0].forward(&path2, Some(&layer1), None)?;
            let chunk = self.output1.forward(&path1, 1, 1);
            let chunk = upsample(
                &chunk,
                (patch_h * self.patch_size, patch_w * self.patch_size),
            )?;
            let chunk = mlxcel_core::astype(&chunk, mlxcel_core::dtype::FLOAT32);
            let chunk = self.output2_first.forward(&chunk, 1, 1);
            let chunk = self.output2_last.forward(&mlxcel_core::relu(&chunk), 1, 0);
            let chunk = mlxcel_core::relu(&chunk);
            mlxcel_core::eval(&chunk);
            chunks.push(chunk);
        }
        let refs: Vec<&MlxArray> = chunks
            .iter()
            .map(|chunk| chunk.as_ref().expect("MLX output chunk"))
            .collect();
        Ok(mlxcel_core::concatenate_many(&refs, 0))
    }
}
