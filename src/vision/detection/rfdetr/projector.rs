// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! RF-DETR C2f multi-feature projector.

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::layers::{ConvNorm, concat, load_norm};

struct Bottleneck {
    first: ConvNorm,
    second: ConvNorm,
}

impl Bottleneck {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            first: ConvNorm::from_weights(weights, &format!("{prefix}.cv1"), 128, 3)?,
            second: ConvNorm::from_weights(weights, &format!("{prefix}.cv2"), 128, 3)?,
        })
    }

    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        self.second.forward(&self.first.forward(input))
    }
}

pub struct Projector {
    first: ConvNorm,
    bottlenecks: Vec<Bottleneck>,
    second: ConvNorm,
    norm: LayerNorm,
}

impl Projector {
    pub fn from_weights(weights: &WeightMap) -> Result<Self, String> {
        let prefix = "projector.stages.0.0";
        Ok(Self {
            first: ConvNorm::from_weights(weights, &format!("{prefix}.cv1"), 256, 1)?,
            bottlenecks: (0..3)
                .map(|index| Bottleneck::from_weights(weights, &format!("{prefix}.m.{index}")))
                .collect::<Result<_, _>>()?,
            second: ConvNorm::from_weights(weights, &format!("{prefix}.cv2"), 256, 1)?,
            norm: load_norm(weights, "projector.stages.0.1", 1.0e-5)?,
        })
    }

    pub fn forward(&self, features: &[UniquePtr<MlxArray>]) -> Result<UniquePtr<MlxArray>, String> {
        if features.len() != 4 {
            return Err(format!(
                "RF-DETR projector expects four backbone features, got {}",
                features.len()
            ));
        }
        let joined = concat(
            &features
                .iter()
                .map(|feature| mlxcel_core::copy(feature))
                .collect::<Vec<_>>(),
            -1,
        );
        let first = self.first.forward(&joined);
        let shape = mlxcel_core::array_shape(&first);
        let half = shape[3] / 2;
        let mut current = mlxcel_core::slice(
            &first,
            &[0, 0, 0, half],
            &[shape[0], shape[1], shape[2], shape[3]],
        );
        let mut branches = vec![first];
        for bottleneck in &self.bottlenecks {
            current = bottleneck.forward(&current);
            branches.push(mlxcel_core::copy(&current));
        }
        Ok(self
            .norm
            .forward(&self.second.forward(&concat(&branches, -1))))
    }
}
