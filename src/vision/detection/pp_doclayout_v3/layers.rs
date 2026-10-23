// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Small inference layers shared by the PP-DocLayoutV3 graph.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub use crate::vision::detection::rt_detr_v2::layers::Activation;

pub fn weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("PP-DocLayoutV3 weight not found: {key}"))
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
        bias: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: bias
                .then(|| weight(weights, &format!("{prefix}.bias")))
                .transpose()?,
            stride: checked_i32(stride, "convolution stride")?,
            padding: checked_i32(padding, "convolution padding")?,
            groups: checked_i32(groups, "convolution groups")?,
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

pub struct BatchNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    mean: UniquePtr<MlxArray>,
    variance: UniquePtr<MlxArray>,
    eps: f32,
}

impl BatchNorm {
    pub fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weight(weights, &format!("{prefix}.bias"))?,
            mean: weight(weights, &format!("{prefix}.running_mean"))?,
            variance: weight(weights, &format!("{prefix}.running_var"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let centered = mlxcel_core::subtract(x, &self.mean);
        let eps = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::array_dtype(x));
        let inv_std = mlxcel_core::rsqrt(&mlxcel_core::add(&self.variance, &eps));
        let normalized = mlxcel_core::multiply(&centered, &inv_std);
        mlxcel_core::add(
            &mlxcel_core::multiply(&normalized, &self.weight),
            &self.bias,
        )
    }
}

pub struct ConvNorm {
    conv: Conv2d,
    bn: BatchNorm,
    activation: Activation,
}

impl ConvNorm {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: usize,
        padding: usize,
        groups: usize,
        activation: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            conv: Conv2d::from_weights(
                weights,
                &format!("{prefix}.conv"),
                stride,
                padding,
                groups,
                false,
            )?,
            bn: BatchNorm::from_weights(weights, &format!("{prefix}.bn"), eps)?,
            activation,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.activation
            .apply(&self.bn.forward(&self.conv.forward(x)))
    }
}

pub fn max_pool_2x2_s1(x: &MlxArray) -> UniquePtr<MlxArray> {
    crate::vision::detection::rt_detr_v2::layers::max_pool2d(x, 2, 1, 0)
}

pub fn upsample_nearest_2x(x: &MlxArray) -> UniquePtr<MlxArray> {
    crate::vision::detection::rt_detr_v2::layers::upsample_nearest_2x(x)
}

pub fn upsample_bilinear_2x(x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(x);
    crate::vision::dense::interpolate::resize_bilinear_nhwc(
        x,
        ((shape[1] * 2) as usize, (shape[2] * 2) as usize),
        false,
        false,
    )
    .map_err(|error| error.to_string())
}

fn checked_i32(value: usize, name: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{name} exceeds the i32 runtime limit"))
}
