// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Sapiens2 deconvolution and pixel-shuffle task heads.

use anyhow::{Result, ensure};
use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{HeadConfig, Sapiens2Config, Sapiens2Task};
use super::weights;

struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    kernel: usize,
    stride: usize,
}

impl Conv2d {
    fn from_weights(
        weights_map: &WeightMap,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: weights::conv_weight(
                weights_map,
                &format!("{prefix}.weight"),
                output,
                kernel,
                input,
            )?,
            bias: weights::array(weights_map, &format!("{prefix}.bias"))?,
            kernel,
            stride,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let padding = (self.kernel / 2) as i32;
        mlxcel_core::add(
            &mlxcel_core::conv2d(
                x,
                &self.weight,
                self.stride as i32,
                self.stride as i32,
                padding,
                padding,
                1,
                1,
                1,
            ),
            &self.bias,
        )
    }
}

struct ConvBlock(Conv2d);

impl ConvBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::silu(&instance_norm(&self.0.forward(x)))
    }
}

struct DeconvBlock {
    weight: UniquePtr<MlxArray>,
    kernel: usize,
}

impl DeconvBlock {
    fn from_weights(
        weights_map: &WeightMap,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(kernel, 2..=4),
            "unsupported Sapiens2 deconvolution kernel {kernel}"
        );
        Ok(Self {
            weight: weights::deconv_weight(
                weights_map,
                &format!("{prefix}.weight"),
                input,
                output,
                kernel,
            )?,
            kernel,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (padding, output_padding) = match self.kernel {
            4 => (1, 0),
            3 => (1, 1),
            2 => (0, 0),
            _ => unreachable!("validated in constructor"),
        };
        let output = mlxcel_core::conv_transpose2d(
            x,
            &self.weight,
            2,
            2,
            padding,
            padding,
            1,
            1,
            output_padding,
            output_padding,
            1,
        );
        mlxcel_core::silu(&instance_norm(&output))
    }
}

fn instance_norm(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let flat = mlxcel_core::reshape(x, &[shape[0], shape[1] * shape[2], shape[3]]);
    let mean = mlxcel_core::mean_axis(&flat, 1, true);
    let variance = mlxcel_core::var_axis(&flat, 1, true, 0);
    let eps = mlxcel_core::full_f32(&[1], 1e-5, mlxcel_core::array_dtype(x));
    let normalized = mlxcel_core::multiply(
        &mlxcel_core::subtract(&flat, &mean),
        &mlxcel_core::rsqrt(&mlxcel_core::add(&variance, &eps)),
    );
    mlxcel_core::reshape(&normalized, &shape)
}

fn pixel_shuffle_2x(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let channels = shape[3] / 4;
    let expanded = mlxcel_core::reshape(x, &[shape[0], shape[1], shape[2], channels, 2, 2]);
    let expanded = mlxcel_core::transpose_axes(&expanded, &[0, 1, 4, 2, 5, 3]);
    mlxcel_core::reshape(&expanded, &[shape[0], shape[1] * 2, shape[2] * 2, channels])
}

struct DeconvHead {
    deconvs: Vec<DeconvBlock>,
    convs: Vec<ConvBlock>,
    predictor: Conv2d,
}

impl DeconvHead {
    fn from_weights(
        weights_map: &WeightMap,
        config: &Sapiens2Config,
        head: &HeadConfig,
    ) -> Result<Self> {
        ensure!(
            head.upsample_out_channels.len() == head.upsample_kernel_sizes.len(),
            "Sapiens2 deconvolution channel/kernel lists differ in length"
        );
        ensure!(
            head.conv_out_channels.len() == head.conv_kernel_sizes.len(),
            "Sapiens2 head convolution channel/kernel lists differ in length"
        );
        let mut channels = config.hidden_size;
        let mut deconvs = Vec::with_capacity(head.upsample_out_channels.len());
        for (index, (&output, &kernel)) in head
            .upsample_out_channels
            .iter()
            .zip(&head.upsample_kernel_sizes)
            .enumerate()
        {
            deconvs.push(DeconvBlock::from_weights(
                weights_map,
                &format!("decode_head.deconv_layers.{}", index * 3),
                channels,
                output,
                kernel,
            )?);
            channels = output;
        }
        let mut convs = Vec::with_capacity(head.conv_out_channels.len());
        for (index, (&output, &kernel)) in head
            .conv_out_channels
            .iter()
            .zip(&head.conv_kernel_sizes)
            .enumerate()
        {
            convs.push(ConvBlock(Conv2d::from_weights(
                weights_map,
                &format!("decode_head.conv_layers.{}", index * 3),
                channels,
                output,
                kernel,
                1,
            )?));
            channels = output;
        }
        let predictor = Conv2d::from_weights(
            weights_map,
            &format!("decode_head.{}", predictor_name(config.task)),
            channels,
            config.num_labels,
            1,
            1,
        )?;
        Ok(Self {
            deconvs,
            convs,
            predictor,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut hidden = mlxcel_core::copy(x);
        for layer in &self.deconvs {
            hidden = layer.forward(&hidden);
        }
        for layer in &self.convs {
            hidden = layer.forward(&hidden);
        }
        self.predictor.forward(&hidden)
    }
}

struct UpsampleBlock {
    convolution: Conv2d,
}

impl UpsampleBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::silu(&instance_norm(&pixel_shuffle_2x(
            &self.convolution.forward(x),
        )))
    }
}

struct PixelShuffleHead {
    input: ConvBlock,
    upsample: Vec<UpsampleBlock>,
    convs: Vec<ConvBlock>,
    predictor: Conv2d,
    scale_convs: Vec<ConvBlock>,
    scale_linears: Vec<Linear>,
}

impl PixelShuffleHead {
    fn from_weights(
        weights_map: &WeightMap,
        config: &Sapiens2Config,
        head: &HeadConfig,
    ) -> Result<Self> {
        ensure!(
            head.conv_out_channels.len() == head.conv_kernel_sizes.len(),
            "Sapiens2 head convolution channel/kernel lists differ in length"
        );
        ensure!(
            head.scale_conv_out_channels.len() == head.scale_conv_kernel_sizes.len(),
            "Sapiens2 scale convolution channel/kernel lists differ in length"
        );
        let input = ConvBlock(Conv2d::from_weights(
            weights_map,
            "decode_head.input_conv.0",
            config.hidden_size,
            config.hidden_size,
            3,
            1,
        )?);
        let mut channels = config.hidden_size;
        let mut upsample = Vec::with_capacity(head.upsample_out_channels.len());
        for (index, &output) in head.upsample_out_channels.iter().enumerate() {
            upsample.push(UpsampleBlock {
                convolution: Conv2d::from_weights(
                    weights_map,
                    &format!("decode_head.upsample_blocks.{index}.0"),
                    channels,
                    output * 4,
                    3,
                    1,
                )?,
            });
            channels = output;
        }
        let mut convs = Vec::with_capacity(head.conv_out_channels.len());
        for (index, (&output, &kernel)) in head
            .conv_out_channels
            .iter()
            .zip(&head.conv_kernel_sizes)
            .enumerate()
        {
            convs.push(ConvBlock(Conv2d::from_weights(
                weights_map,
                &format!("decode_head.conv_layers.{}", index * 3),
                channels,
                output,
                kernel,
                1,
            )?));
            channels = output;
        }
        let predictor = Conv2d::from_weights(
            weights_map,
            &format!("decode_head.{}", predictor_name(config.task)),
            channels,
            config.num_labels,
            1,
            1,
        )?;

        let mut scale_channels = config.hidden_size;
        let mut scale_convs = Vec::with_capacity(head.scale_conv_out_channels.len());
        for (index, (&output, &kernel)) in head
            .scale_conv_out_channels
            .iter()
            .zip(&head.scale_conv_kernel_sizes)
            .enumerate()
        {
            scale_convs.push(ConvBlock(Conv2d::from_weights(
                weights_map,
                &format!("decode_head.scale_conv_layers.{}", index * 3),
                scale_channels,
                output,
                kernel,
                2,
            )?));
            scale_channels = output;
        }
        let mut scale_linears = Vec::new();
        if let Some(input_size) = head.scale_final_input_size {
            let mut dims = vec![input_size];
            dims.extend_from_slice(&head.scale_final_hidden_sizes);
            dims.push(1);
            for index in 0..dims.len() - 1 {
                scale_linears.push(weights::linear(
                    weights_map,
                    &format!("decode_head.scale_final_layer.{}", index * 2 + 1),
                )?);
            }
        }
        Ok(Self {
            input,
            upsample,
            convs,
            predictor,
            scale_convs,
            scale_linears,
        })
    }

    fn forward(&self, x: &MlxArray) -> HeadOutput {
        let mut output = self.input.forward(x);
        for layer in &self.upsample {
            output = layer.forward(&output);
        }
        for layer in &self.convs {
            output = layer.forward(&output);
        }
        output = self.predictor.forward(&output);

        let scale = if self.scale_convs.is_empty() {
            None
        } else {
            let mut hidden = mlxcel_core::copy(x);
            for layer in &self.scale_convs {
                hidden = layer.forward(&hidden);
            }
            let shape = mlxcel_core::array_shape(&hidden);
            hidden = mlxcel_core::reshape(
                &mlxcel_core::transpose_axes(&hidden, &[0, 3, 1, 2]),
                &[shape[0], shape[1] * shape[2] * shape[3]],
            );
            for (index, linear) in self.scale_linears.iter().enumerate() {
                hidden = linear.forward(&hidden);
                if index + 1 < self.scale_linears.len() {
                    hidden = mlxcel_core::silu(&hidden);
                }
            }
            Some(hidden)
        };
        HeadOutput {
            prediction: output,
            scale,
        }
    }
}

pub struct HeadOutput {
    pub prediction: UniquePtr<MlxArray>,
    pub scale: Option<UniquePtr<MlxArray>>,
}

pub enum Sapiens2Head {
    Deconv(DeconvHead),
    PixelShuffle(PixelShuffleHead),
}

impl Sapiens2Head {
    pub fn from_weights(weights_map: &WeightMap, config: &Sapiens2Config) -> Result<Option<Self>> {
        let Some(head) = &config.head else {
            return Ok(None);
        };
        if head.use_pixel_shuffle {
            Ok(Some(Self::PixelShuffle(PixelShuffleHead::from_weights(
                weights_map,
                config,
                head,
            )?)))
        } else {
            Ok(Some(Self::Deconv(DeconvHead::from_weights(
                weights_map,
                config,
                head,
            )?)))
        }
    }

    pub fn forward(&self, x: &MlxArray) -> HeadOutput {
        match self {
            Self::Deconv(head) => HeadOutput {
                prediction: head.forward(x),
                scale: None,
            },
            Self::PixelShuffle(head) => head.forward(x),
        }
    }
}

fn predictor_name(task: Sapiens2Task) -> &'static str {
    match task {
        Sapiens2Task::Pose => "conv_pose",
        Sapiens2Task::Segmentation => "conv_seg",
        Sapiens2Task::Normal => "conv_normal",
        Sapiens2Task::Pointmap => "conv_pointmap",
        Sapiens2Task::Matting => "conv_matting",
        Sapiens2Task::Backbone => unreachable!("backbone has no decode head"),
    }
}
