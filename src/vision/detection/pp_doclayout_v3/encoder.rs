// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! PP-DocLayoutV3 hybrid AIFI/FPN/PAN encoder.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::aifi::Aifi;
use super::config::PpDocLayoutV3Config;
use super::layers::{Activation, ConvNorm, upsample_nearest_2x};
use super::mask::MaskFeaturePath;

struct RepVgg {
    spatial: ConvNorm,
    pointwise: ConvNorm,
    activation: Activation,
}

impl RepVgg {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        activation: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            spatial: ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv1"),
                1,
                1,
                1,
                Activation::None,
                eps,
            )?,
            pointwise: ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv2"),
                1,
                0,
                1,
                Activation::None,
                eps,
            )?,
            activation,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.activation.apply(&mlxcel_core::add(
            &self.spatial.forward(x),
            &self.pointwise.forward(x),
        ))
    }
}

struct CspRep {
    first: ConvNorm,
    second: ConvNorm,
    bottlenecks: Vec<RepVgg>,
    output: Option<ConvNorm>,
}

impl CspRep {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        output_channels: usize,
        hidden_expansion: f32,
        activation: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        let hidden = (output_channels as f32 * hidden_expansion) as usize;
        let first = ConvNorm::from_weights(
            weights,
            &format!("{prefix}.conv1"),
            1,
            0,
            1,
            activation,
            eps,
        )?;
        let second = ConvNorm::from_weights(
            weights,
            &format!("{prefix}.conv2"),
            1,
            0,
            1,
            activation,
            eps,
        )?;
        let mut bottlenecks = Vec::with_capacity(3);
        for index in 0..3 {
            bottlenecks.push(RepVgg::from_weights(
                weights,
                &format!("{prefix}.bottlenecks.{index}"),
                activation,
                eps,
            )?);
        }
        let output = (hidden != output_channels)
            .then(|| {
                ConvNorm::from_weights(
                    weights,
                    &format!("{prefix}.conv3"),
                    1,
                    0,
                    1,
                    activation,
                    eps,
                )
            })
            .transpose()?;
        Ok(Self {
            first,
            second,
            bottlenecks,
            output,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut first = self.first.forward(x);
        for block in &self.bottlenecks {
            first = block.forward(&first);
        }
        let joined = mlxcel_core::add(&first, &self.second.forward(x));
        match &self.output {
            Some(layer) => layer.forward(&joined),
            None => joined,
        }
    }
}

pub struct HybridEncoder {
    aifi: Aifi,
    encode_level: usize,
    lateral: Vec<ConvNorm>,
    fpn: Vec<CspRep>,
    downsample: Vec<ConvNorm>,
    pan: Vec<CspRep>,
    mask: MaskFeaturePath,
}

impl HybridEncoder {
    pub fn from_weights(weights: &WeightMap, config: &PpDocLayoutV3Config) -> Result<Self, String> {
        let activation = Activation::parse(&config.activation_function)?;
        let eps = config.batch_norm_eps;
        let transitions = config.encoder_in_channels.len() - 1;
        let mut lateral = Vec::with_capacity(transitions);
        let mut fpn = Vec::with_capacity(transitions);
        let mut downsample = Vec::with_capacity(transitions);
        let mut pan = Vec::with_capacity(transitions);
        for index in 0..transitions {
            lateral.push(ConvNorm::from_weights(
                weights,
                &format!("encoder.lateral_convs.{index}"),
                1,
                0,
                1,
                activation,
                eps,
            )?);
            fpn.push(CspRep::from_weights(
                weights,
                &format!("encoder.fpn_blocks.{index}"),
                config.encoder_hidden_dim,
                config.hidden_expansion,
                activation,
                eps,
            )?);
            downsample.push(ConvNorm::from_weights(
                weights,
                &format!("encoder.downsample_convs.{index}"),
                2,
                1,
                1,
                activation,
                eps,
            )?);
            pan.push(CspRep::from_weights(
                weights,
                &format!("encoder.pan_blocks.{index}"),
                config.encoder_hidden_dim,
                config.hidden_expansion,
                activation,
                eps,
            )?);
        }
        Ok(Self {
            aifi: Aifi::from_weights(weights, "encoder.aifi.0", config)?,
            encode_level: config.encode_proj_layers[0],
            lateral,
            fpn,
            downsample,
            pan,
            mask: MaskFeaturePath::from_weights(weights, config)?,
        })
    }

    pub fn forward(
        &self,
        mut features: Vec<UniquePtr<MlxArray>>,
        stride_four: &MlxArray,
    ) -> Result<(Vec<UniquePtr<MlxArray>>, UniquePtr<MlxArray>), String> {
        features[self.encode_level] = self.aifi.forward(&features[self.encode_level]);

        let mut fpn = vec![mlxcel_core::copy(
            features.last().expect("validated feature levels"),
        )];
        for (index, (lateral, block)) in self.lateral.iter().zip(&self.fpn).enumerate() {
            let top = lateral.forward(fpn.last().expect("FPN seed"));
            *fpn.last_mut().expect("FPN seed") = mlxcel_core::copy(&top);
            let upsampled = upsample_nearest_2x(&top);
            let source = &features[features.len() - 2 - index];
            fpn.push(block.forward(&mlxcel_core::concatenate(&upsampled, source, -1)));
        }
        fpn.reverse();

        let mut pan = vec![mlxcel_core::copy(&fpn[0])];
        for (index, (downsample, block)) in self.downsample.iter().zip(&self.pan).enumerate() {
            let lower = downsample.forward(pan.last().expect("PAN seed"));
            pan.push(block.forward(&mlxcel_core::concatenate(&lower, &fpn[index + 1], -1)));
        }
        #[cfg(test)]
        for (index, feature) in pan.iter().enumerate() {
            super::diagnostics::capture(&format!("encoder.pan{index}"), feature);
        }
        let mask = self.mask.forward(&pan, stride_four)?;
        #[cfg(test)]
        super::diagnostics::capture("encoder.mask_feature", &mask);
        Ok((pan, mask))
    }
}
