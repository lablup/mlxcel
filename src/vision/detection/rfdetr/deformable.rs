// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! One-level deformable cross-attention for RF-DETR.

use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::detection::pp_doclayout_v3::deformable::grid_sample_f32;

pub struct DeformableAttention {
    offsets: Linear,
    attention: Linear,
    value: Linear,
    output: Linear,
    heads: i32,
    points: i32,
    head_dim: i32,
}

impl DeformableAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        heads: usize,
        points: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            offsets: Linear::from_weights(weights, &format!("{prefix}.sampling_offsets"))?,
            attention: Linear::from_weights(weights, &format!("{prefix}.attention_weights"))?,
            value: Linear::from_weights(weights, &format!("{prefix}.value_proj"))?,
            output: Linear::from_weights(weights, &format!("{prefix}.output_proj"))?,
            heads: heads as i32,
            points: points as i32,
            head_dim: 256 / heads as i32,
        })
    }

    pub fn forward(
        &self,
        query: &MlxArray,
        reference: &MlxArray,
        memory: &MlxArray,
        spatial: (i32, i32),
    ) -> UniquePtr<MlxArray> {
        let query_shape = mlxcel_core::array_shape(query);
        let (batch, queries) = (query_shape[0], query_shape[1]);
        let (height, width) = spatial;
        let length = height * width;
        let value = mlxcel_core::reshape(
            &self.value.forward(memory),
            &[batch, length, self.heads, self.head_dim],
        );
        let offsets = mlxcel_core::reshape(
            &self.offsets.forward(query),
            &[batch, queries, self.heads, self.points, 2],
        );
        let attention = mlxcel_core::softmax(
            &mlxcel_core::reshape(
                &self.attention.forward(query),
                &[batch, queries, self.heads, self.points],
            ),
            -1,
        );
        let ref_xy = mlxcel_core::slice(reference, &[0, 0, 0, 0], &[batch, queries, 1, 2]);
        let ref_wh = mlxcel_core::slice(reference, &[0, 0, 0, 2], &[batch, queries, 1, 4]);
        let ref_xy = mlxcel_core::expand_dims(&ref_xy, 2);
        let ref_wh = mlxcel_core::expand_dims(&ref_wh, 2);
        let offsets = mlxcel_core::multiply_scalar(&offsets, 0.5 / self.points as f32);
        let locations = mlxcel_core::add(&ref_xy, &mlxcel_core::multiply(&offsets, &ref_wh));

        let value =
            mlxcel_core::reshape(&value, &[batch, height, width, self.heads, self.head_dim]);
        let value = mlxcel_core::transpose_axes(&value, &[0, 3, 1, 2, 4]);
        let value =
            mlxcel_core::reshape(&value, &[batch * self.heads, height, width, self.head_dim]);
        let grid = mlxcel_core::transpose_axes(&locations, &[0, 2, 1, 3, 4]);
        let grid = mlxcel_core::reshape(&grid, &[batch * self.heads, queries, self.points, 2]);
        let grid = mlxcel_core::subtract(
            &mlxcel_core::multiply_scalar(&grid, 2.0),
            &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&grid)),
        );
        let sampled = grid_sample_f32(&value, &grid);
        let weights = mlxcel_core::transpose_axes(&attention, &[0, 2, 1, 3]);
        let weights =
            mlxcel_core::reshape(&weights, &[batch * self.heads, queries, self.points, 1]);
        let output = mlxcel_core::sum_axis(&mlxcel_core::multiply(&sampled, &weights), 2, false);
        let output = mlxcel_core::reshape(&output, &[batch, self.heads, queries, self.head_dim]);
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        self.output
            .forward(&mlxcel_core::reshape(&output, &[batch, queries, 256]))
    }
}
