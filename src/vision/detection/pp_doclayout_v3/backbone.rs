// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! HGNetV2-L backbone used by PP-DocLayoutV3.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::HgNetV2Config;
use super::layers::{Activation, ConvNorm, max_pool_2x2_s1};

struct LightConv {
    pointwise: ConvNorm,
    depthwise: ConvNorm,
}

impl LightConv {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        kernel: usize,
        activation: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            pointwise: ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv1"),
                1,
                0,
                1,
                Activation::None,
                eps,
            )?,
            depthwise: ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv2"),
                1,
                kernel / 2,
                channels,
                activation,
                eps,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.depthwise.forward(&self.pointwise.forward(x))
    }
}

enum BasicConv {
    Regular(ConvNorm),
    Light(LightConv),
}

impl BasicConv {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Regular(layer) => layer.forward(x),
            Self::Light(layer) => layer.forward(x),
        }
    }
}

struct BasicLayer {
    layers: Vec<BasicConv>,
    aggregation: [ConvNorm; 2],
    residual: bool,
}

impl BasicLayer {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _input: usize,
        middle: usize,
        count: usize,
        kernel: usize,
        residual: bool,
        light: bool,
        activation: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(count);
        for index in 0..count {
            let layer_prefix = format!("{prefix}.layers.{index}");
            layers.push(if light {
                BasicConv::Light(LightConv::from_weights(
                    weights,
                    &layer_prefix,
                    middle,
                    kernel,
                    activation,
                    eps,
                )?)
            } else {
                BasicConv::Regular(ConvNorm::from_weights(
                    weights,
                    &layer_prefix,
                    1,
                    kernel / 2,
                    1,
                    activation,
                    eps,
                )?)
            });
        }
        Ok(Self {
            layers,
            aggregation: [
                ConvNorm::from_weights(
                    weights,
                    &format!("{prefix}.aggregation.0"),
                    1,
                    0,
                    1,
                    activation,
                    eps,
                )?,
                ConvNorm::from_weights(
                    weights,
                    &format!("{prefix}.aggregation.1"),
                    1,
                    0,
                    1,
                    activation,
                    eps,
                )?,
            ],
            residual,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let identity = mlxcel_core::copy(x);
        let mut current = mlxcel_core::copy(x);
        let mut features = vec![mlxcel_core::copy(x)];
        for layer in &self.layers {
            current = layer.forward(&current);
            features.push(mlxcel_core::copy(&current));
        }
        let refs: Vec<&MlxArray> = features
            .iter()
            .map(|value| value.as_ref().expect("MLX array allocation"))
            .collect();
        let mut output = mlxcel_core::concatenate_many(&refs, -1);
        output = self.aggregation[0].forward(&output);
        output = self.aggregation[1].forward(&output);
        if self.residual {
            mlxcel_core::add(&output, &identity)
        } else {
            output
        }
    }
}

struct Stage {
    downsample: Option<ConvNorm>,
    blocks: Vec<BasicLayer>,
}

impl Stage {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        input: usize,
        middle: usize,
        output: usize,
        blocks: usize,
        layers: usize,
        downsample: bool,
        downsample_stride: usize,
        light: bool,
        kernel: usize,
        activation: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        let downsample = downsample
            .then(|| {
                ConvNorm::from_weights(
                    weights,
                    &format!("{prefix}.downsample"),
                    downsample_stride,
                    1,
                    input,
                    Activation::None,
                    eps,
                )
            })
            .transpose()?;
        let mut built = Vec::with_capacity(blocks);
        for index in 0..blocks {
            built.push(BasicLayer::from_weights(
                weights,
                &format!("{prefix}.blocks.{index}"),
                if index == 0 { input } else { output },
                middle,
                layers,
                kernel,
                index != 0,
                light,
                activation,
                eps,
            )?);
        }
        Ok(Self {
            downsample,
            blocks: built,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut output = match &self.downsample {
            Some(layer) => layer.forward(x),
            None => mlxcel_core::copy(x),
        };
        for block in &self.blocks {
            output = block.forward(&output);
        }
        output
    }
}

struct Stem {
    layers: [ConvNorm; 5],
}

impl Stem {
    fn from_weights(weights: &WeightMap, config: &HgNetV2Config, eps: f32) -> Result<Self, String> {
        let act = Activation::parse(&config.hidden_act)?;
        Ok(Self {
            layers: [
                ConvNorm::from_weights(weights, "backbone.embedder.stem1", 2, 1, 1, act, eps)?,
                ConvNorm::from_weights(weights, "backbone.embedder.stem2a", 1, 0, 1, act, eps)?,
                ConvNorm::from_weights(weights, "backbone.embedder.stem2b", 1, 0, 1, act, eps)?,
                ConvNorm::from_weights(weights, "backbone.embedder.stem3", 2, 1, 1, act, eps)?,
                ConvNorm::from_weights(weights, "backbone.embedder.stem4", 1, 0, 1, act, eps)?,
            ],
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let stem1 = self.layers[0].forward(x);
        let padded = mlxcel_core::pad(&stem1, &[0, 0, 0, 1, 0, 1, 0, 0], 0.0);
        let a = self.layers[1].forward(&padded);
        let a = mlxcel_core::pad(&a, &[0, 0, 0, 1, 0, 1, 0, 0], 0.0);
        let a = self.layers[2].forward(&a);
        let pooled = max_pool_2x2_s1(&padded);
        let joined = mlxcel_core::concatenate(&pooled, &a, -1);
        self.layers[4].forward(&self.layers[3].forward(&joined))
    }
}

pub struct HgNetV2Backbone {
    stem: Stem,
    stages: Vec<Stage>,
}

impl HgNetV2Backbone {
    pub fn from_weights(
        weights: &WeightMap,
        config: &HgNetV2Config,
        eps: f32,
    ) -> Result<Self, String> {
        let activation = Activation::parse(&config.hidden_act)?;
        let mut stages = Vec::with_capacity(4);
        for index in 0..4 {
            stages.push(Stage::from_weights(
                weights,
                &format!("backbone.encoder.stages.{index}"),
                config.stage_in_channels[index],
                config.stage_mid_channels[index],
                config.stage_out_channels[index],
                config.stage_num_blocks[index],
                config.stage_numb_of_layers[index],
                config.stage_downsample[index],
                config.stage_downsample_strides[index],
                config.stage_light_block[index],
                config.stage_kernel_size[index],
                activation,
                eps,
            )?);
        }
        Ok(Self {
            stem: Stem::from_weights(weights, config, eps)?,
            stages,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        let mut current = self.stem.forward(x);
        #[cfg(test)]
        super::diagnostics::capture("backbone.stem", &current);
        let mut features = Vec::with_capacity(self.stages.len());
        for (_index, stage) in self.stages.iter().enumerate() {
            current = stage.forward(&current);
            #[cfg(test)]
            super::diagnostics::capture(&format!("backbone.stage{_index}"), &current);
            features.push(mlxcel_core::copy(&current));
        }
        features
    }
}
