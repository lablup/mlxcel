// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Mask-feature pyramid used for mask-enhanced query initialization.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::PpDocLayoutV3Config;
use super::layers::{Activation, Conv2d, ConvNorm, upsample_bilinear_2x};

struct ScaleHead {
    convolutions: Vec<(usize, ConvNorm)>,
    upsample: bool,
}

impl ScaleHead {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        convolutions: usize,
        upsample: bool,
        eps: f32,
    ) -> Result<Self, String> {
        let mut built = Vec::with_capacity(convolutions);
        for index in 0..convolutions {
            let key_index = if upsample { index * 2 } else { index };
            built.push((
                key_index,
                ConvNorm::from_weights(
                    weights,
                    &format!("{prefix}.layers.{key_index}"),
                    1,
                    1,
                    1,
                    Activation::Silu,
                    eps,
                )?,
            ));
        }
        Ok(Self {
            convolutions: built,
            upsample,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let mut output = mlxcel_core::copy(x);
        for (_, convolution) in &self.convolutions {
            output = convolution.forward(&output);
            if self.upsample {
                output = upsample_bilinear_2x(&output)?;
            }
        }
        Ok(output)
    }
}

pub struct MaskFeaturePath {
    scale_heads: Vec<ScaleHead>,
    output_conv: ConvNorm,
    lateral: ConvNorm,
    base_conv: ConvNorm,
    prototype_conv: Conv2d,
}

impl MaskFeaturePath {
    pub fn from_weights(weights: &WeightMap, config: &PpDocLayoutV3Config) -> Result<Self, String> {
        let eps = config.batch_norm_eps;
        let base_stride = *config
            .feat_strides
            .iter()
            .min()
            .ok_or_else(|| "PP-DocLayoutV3 feat_strides cannot be empty".to_string())?;
        let mut scale_heads = Vec::with_capacity(config.feat_strides.len());
        for (index, &stride) in config.feat_strides.iter().enumerate() {
            let ratio = stride / base_stride;
            let log = usize::BITS - ratio.leading_zeros() - 1;
            let count = 1usize.max(log as usize);
            scale_heads.push(ScaleHead::from_weights(
                weights,
                &format!("encoder.mask_feature_head.scale_heads.{index}"),
                count,
                stride != base_stride,
                eps,
            )?);
        }
        Ok(Self {
            scale_heads,
            output_conv: ConvNorm::from_weights(
                weights,
                "encoder.mask_feature_head.output_conv",
                1,
                1,
                1,
                Activation::Silu,
                eps,
            )?,
            lateral: ConvNorm::from_weights(
                weights,
                "encoder.encoder_mask_lateral",
                1,
                1,
                1,
                Activation::Silu,
                eps,
            )?,
            base_conv: ConvNorm::from_weights(
                weights,
                "encoder.encoder_mask_output.base_conv",
                1,
                1,
                1,
                Activation::Silu,
                eps,
            )?,
            prototype_conv: Conv2d::from_weights(
                weights,
                "encoder.encoder_mask_output.conv",
                1,
                0,
                1,
                true,
            )?,
        })
    }

    pub fn forward(
        &self,
        pan: &[UniquePtr<MlxArray>],
        stride_four: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let mut fused = self.scale_heads[0].forward(&pan[0])?;
        for (head, feature) in self.scale_heads.iter().skip(1).zip(pan.iter().skip(1)) {
            let projected = head.forward(feature)?;
            fused = mlxcel_core::add(&fused, &projected);
        }
        let fused = self.output_conv.forward(&fused);
        let fused = upsample_bilinear_2x(&fused)?;
        let fused = mlxcel_core::add(&fused, &self.lateral.forward(stride_four));
        Ok(self.prototype_conv.forward(&self.base_conv.forward(&fused)))
    }
}
