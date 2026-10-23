// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! PP-DocLayoutV3 multi-scale deformable attention.

use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::PpDocLayoutV3Config;

pub type SpatialShape = (i32, i32);

pub struct MsDeformableAttention {
    offsets: Linear,
    weights: Linear,
    value: Linear,
    output: Linear,
    heads: i32,
    levels: i32,
    points: i32,
    head_dim: i32,
}

impl MsDeformableAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &PpDocLayoutV3Config,
    ) -> Result<Self, String> {
        Ok(Self {
            offsets: Linear::from_weights(weights, &format!("{prefix}.sampling_offsets"))?,
            weights: Linear::from_weights(weights, &format!("{prefix}.attention_weights"))?,
            value: Linear::from_weights(weights, &format!("{prefix}.value_proj"))?,
            output: Linear::from_weights(weights, &format!("{prefix}.output_proj"))?,
            heads: config.decoder_attention_heads as i32,
            levels: config.num_feature_levels as i32,
            points: config.decoder_n_points as i32,
            head_dim: (config.d_model / config.decoder_attention_heads) as i32,
        })
    }

    pub fn forward(
        &self,
        query: &MlxArray,
        reference_points: &MlxArray,
        value: &MlxArray,
        shapes: &[SpatialShape],
        position: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let query = match position {
            Some(position) => mlxcel_core::add(query, position),
            None => mlxcel_core::copy(query),
        };
        let query_shape = mlxcel_core::array_shape(&query);
        let (batch, queries, model_dim) = (query_shape[0], query_shape[1], query_shape[2]);
        let total_points = self.levels * self.points;
        let value_length = mlxcel_core::array_shape(value)[1];
        let value = mlxcel_core::reshape(
            &self.value.forward(value),
            &[batch, value_length, self.heads, self.head_dim],
        );
        let offsets = mlxcel_core::reshape(
            &self.offsets.forward(&query),
            &[batch, queries, self.heads, total_points, 2],
        );
        let attention = mlxcel_core::softmax(
            &mlxcel_core::reshape(
                &self.weights.forward(&query),
                &[batch, queries, self.heads, total_points],
            ),
            -1,
        );

        let ref_xy = mlxcel_core::slice(reference_points, &[0, 0, 0, 0], &[batch, queries, 1, 2]);
        let ref_wh = mlxcel_core::slice(reference_points, &[0, 0, 0, 2], &[batch, queries, 1, 4]);
        let ref_xy = mlxcel_core::expand_dims(&ref_xy, 2);
        let ref_wh = mlxcel_core::expand_dims(&ref_wh, 2);
        let scaled = mlxcel_core::multiply_scalar(&offsets, 0.5 / self.points as f32);
        let locations = mlxcel_core::add(&ref_xy, &mlxcel_core::multiply(&scaled, &ref_wh));

        let mut sampled = Vec::with_capacity(shapes.len());
        let mut value_offset = 0;
        for (level, &(height, width)) in shapes.iter().enumerate() {
            let level_size = height * width;
            let level_value = mlxcel_core::slice(
                &value,
                &[0, value_offset, 0, 0],
                &[batch, value_offset + level_size, self.heads, self.head_dim],
            );
            value_offset += level_size;
            let level_value = mlxcel_core::reshape(
                &level_value,
                &[batch, height, width, self.heads, self.head_dim],
            );
            let level_value = mlxcel_core::transpose_axes(&level_value, &[0, 3, 1, 2, 4]);
            let level_value = mlxcel_core::reshape(
                &level_value,
                &[batch * self.heads, height, width, self.head_dim],
            );

            let start = level as i32 * self.points;
            let grid = mlxcel_core::slice(
                &locations,
                &[0, 0, 0, start, 0],
                &[batch, queries, self.heads, start + self.points, 2],
            );
            let grid = mlxcel_core::transpose_axes(&grid, &[0, 2, 1, 3, 4]);
            let grid = mlxcel_core::reshape(&grid, &[batch * self.heads, queries, self.points, 2]);
            let grid = mlxcel_core::subtract(
                &mlxcel_core::multiply_scalar(&grid, 2.0),
                &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&grid)),
            );
            sampled.push(grid_sample_f32(&level_value, &grid));
        }
        let refs: Vec<&MlxArray> = sampled
            .iter()
            .map(|value| value.as_ref().expect("MLX array allocation"))
            .collect();
        let sampled = mlxcel_core::concatenate_many(&refs, -2);
        let attention = mlxcel_core::transpose_axes(&attention, &[0, 2, 1, 3]);
        let attention =
            mlxcel_core::reshape(&attention, &[batch * self.heads, queries, total_points]);
        let attention = mlxcel_core::expand_dims(&attention, -1);
        let output = mlxcel_core::sum_axis(&mlxcel_core::multiply(&sampled, &attention), -2, false);
        let output = mlxcel_core::reshape(&output, &[batch, self.heads, queries, self.head_dim]);
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        self.output
            .forward(&mlxcel_core::reshape(&output, &[batch, queries, model_dim]))
    }
}

/// Run the shared bilinear kernel in f32 and return the input feature dtype.
pub fn grid_sample_f32(input: &MlxArray, grid: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(input);
    let input_f32 = mlxcel_core::astype(input, mlxcel_core::dtype::FLOAT32);
    let grid_f32 = mlxcel_core::astype(grid, mlxcel_core::dtype::FLOAT32);
    let output = crate::vision::detection::rt_detr_v2::layers::grid_sample(&input_f32, &grid_f32);
    mlxcel_core::astype(&output, dtype)
}
