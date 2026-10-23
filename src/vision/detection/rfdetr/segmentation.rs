// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Optional RF-DETR instance-segmentation head.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::dense::interpolate::resize_bilinear_nhwc;

use super::layers::{Conv2d, copy_weight, load_norm};

struct DepthwiseBlock {
    convolution: Conv2d,
    norm: LayerNorm,
    projection: Linear,
}

impl DepthwiseBlock {
    fn from_weights(weights: &WeightMap, index: usize) -> Result<Self, String> {
        let prefix = format!("segmentation_head.blocks.{index}");
        Ok(Self {
            convolution: Conv2d::from_weights(weights, &format!("{prefix}.dwconv"), 1, 1, 256)?,
            norm: load_norm(weights, &format!("{prefix}.norm"), 1.0e-6)?,
            projection: Linear::from_weights(weights, &format!("{prefix}.pwconv1"))?,
        })
    }

    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let output = self.convolution.forward(input);
        let output = self.norm.forward(&output);
        let output = mlxcel_core::gelu(&self.projection.forward(&output));
        mlxcel_core::add(input, &output)
    }
}

pub struct SegmentationHead {
    blocks: Vec<DepthwiseBlock>,
    spatial_projection: Conv2d,
    query_norm: LayerNorm,
    query_first: Linear,
    query_second: Linear,
    query_projection: Linear,
    bias: UniquePtr<MlxArray>,
}

impl SegmentationHead {
    pub fn from_weights(weights: &WeightMap, blocks: usize) -> Result<Self, String> {
        Ok(Self {
            blocks: (0..blocks)
                .map(|index| DepthwiseBlock::from_weights(weights, index))
                .collect::<Result<_, _>>()?,
            spatial_projection: Conv2d::from_weights(
                weights,
                "segmentation_head.spatial_features_proj",
                1,
                0,
                1,
            )?,
            query_norm: load_norm(
                weights,
                "segmentation_head.query_features_block.norm_in",
                1.0e-5,
            )?,
            query_first: Linear::from_weights(
                weights,
                "segmentation_head.query_features_block.layers.0",
            )?,
            query_second: Linear::from_weights(
                weights,
                "segmentation_head.query_features_block.layers.2",
            )?,
            query_projection: Linear::from_weights(
                weights,
                "segmentation_head.query_features_proj",
            )?,
            bias: copy_weight(weights, "segmentation_head.bias")?,
        })
    }

    pub fn forward(
        &self,
        spatial: &MlxArray,
        queries: &MlxArray,
        image_size: (usize, usize),
    ) -> Result<UniquePtr<MlxArray>, String> {
        let target = (image_size.0 / 4, image_size.1 / 4);
        let mut spatial = resize_bilinear_nhwc(spatial, target, true, false)
            .map_err(|error| format!("RF-DETR mask feature resize failed: {error}"))?;
        for block in &self.blocks {
            spatial = block.forward(&spatial);
        }
        let spatial = self.spatial_projection.forward(&spatial);
        let query = self.query_norm.forward(queries);
        let query = self.query_first.forward(&query);
        let query = mlxcel_core::gelu(&query);
        let query = mlxcel_core::add(queries, &self.query_second.forward(&query));
        let query = self.query_projection.forward(&query);
        let spatial_ptr = spatial.as_ref().expect("MLX array allocation") as *const MlxArray;
        let query_ptr = query.as_ref().expect("MLX array allocation") as *const MlxArray;
        // SAFETY: both pointers refer to arrays owned above and live for the full call.
        let masks = unsafe { mlxcel_core::einsum("bhwc,bnc->bnhw", &[spatial_ptr, query_ptr]) };
        Ok(mlxcel_core::add(&masks, &self.bias))
    }
}
