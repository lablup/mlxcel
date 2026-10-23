// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Small inference layers used by RF-DETR.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("RF-DETR weight not found: {key}"))
}

pub fn load_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        copy_weight(weights, &format!("{prefix}.weight"))?,
        weights
            .get(&format!("{prefix}.bias"))
            .map(|value| mlxcel_core::copy(value)),
        eps,
    ))
}

pub struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv2d {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .get(&format!("{prefix}.bias"))
                .map(|value| mlxcel_core::copy(value)),
            stride: i32::try_from(stride).map_err(|_| "RF-DETR convolution stride overflow")?,
            padding: i32::try_from(padding).map_err(|_| "RF-DETR convolution padding overflow")?,
            groups: i32::try_from(groups).map_err(|_| "RF-DETR convolution groups overflow")?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let output = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            self.groups,
        );
        match &self.bias {
            Some(bias) => mlxcel_core::add(&output, bias),
            None => output,
        }
    }
}

pub struct ConvNorm {
    conv: Conv2d,
    norm: LayerNorm,
}

impl ConvNorm {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _channels: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            conv: Conv2d::from_weights(weights, &format!("{prefix}.conv"), 1, kernel / 2, 1)?,
            norm: load_norm(weights, &format!("{prefix}.bn"), 1.0e-5)?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::silu(&self.norm.forward(&self.conv.forward(x)))
    }
}

pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
}

#[derive(Clone, Copy)]
pub enum Activation {
    Relu,
}

impl Mlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        count: usize,
        activation: Activation,
    ) -> Result<Self, String> {
        let layers = (0..count)
            .map(|index| Linear::from_weights(weights, &format!("{prefix}.layers.{index}")))
            .collect::<Result<_, _>>()?;
        Ok(Self { layers, activation })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut output = mlxcel_core::copy(x);
        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.forward(&output);
            if index + 1 < self.layers.len() {
                output = match self.activation {
                    Activation::Relu => mlxcel_core::relu(&output),
                };
            }
        }
        output
    }
}

pub fn concat(values: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let refs: Vec<&MlxArray> = values
        .iter()
        .map(|value| value.as_ref().expect("MLX array allocation"))
        .collect();
    mlxcel_core::concatenate_many(&refs, axis)
}
