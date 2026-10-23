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
        _layer_index: usize,
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
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.query", &query);
        }
        let query_shape = mlxcel_core::array_shape(&query);
        let (batch, queries, model_dim) = (query_shape[0], query_shape[1], query_shape[2]);
        let total_points = self.levels * self.points;
        let value_length = mlxcel_core::array_shape(value)[1];
        let value = mlxcel_core::reshape(
            &self.value.forward(value),
            &[batch, value_length, self.heads, self.head_dim],
        );
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.value", &value);
        }
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
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.offsets", &offsets);
            super::diagnostics::capture("decoder.layer0.cross.attention", &attention);
        }

        let ref_xy = mlxcel_core::slice(reference_points, &[0, 0, 0, 0], &[batch, queries, 1, 2]);
        let ref_wh = mlxcel_core::slice(reference_points, &[0, 0, 0, 2], &[batch, queries, 1, 4]);
        let ref_xy = mlxcel_core::expand_dims(&ref_xy, 2);
        let ref_wh = mlxcel_core::expand_dims(&ref_wh, 2);
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.reference_xy", &ref_xy);
            super::diagnostics::capture("decoder.layer0.cross.reference_wh", &ref_wh);
        }
        let scaled = mlxcel_core::multiply_scalar(&offsets, 0.5 / self.points as f32);
        let locations = mlxcel_core::add(&ref_xy, &mlxcel_core::multiply(&scaled, &ref_wh));
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.locations", &locations);
        }

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
            let level_sampled = grid_sample_f32(&level_value, &grid);
            #[cfg(test)]
            if _layer_index == 0 {
                super::diagnostics::capture(
                    &format!("decoder.layer0.cross.sampled.level{level}"),
                    &level_sampled,
                );
            }
            sampled.push(level_sampled);
        }
        let refs: Vec<&MlxArray> = sampled
            .iter()
            .map(|value| value.as_ref().expect("MLX array allocation"))
            .collect();
        let sampled = mlxcel_core::concatenate_many(&refs, -2);
        let weights = mlxcel_core::transpose_axes(&attention, &[0, 2, 1, 3]);
        let weights = mlxcel_core::reshape(&weights, &[batch * self.heads, queries, total_points]);
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.sampled", &sampled);
            super::diagnostics::capture("decoder.layer0.cross.weights", &weights);
        }
        let weights = mlxcel_core::expand_dims(&weights, -1);
        let output = mlxcel_core::sum_axis(&mlxcel_core::multiply(&sampled, &weights), -2, false);
        let output = mlxcel_core::reshape(&output, &[batch, self.heads, queries, self.head_dim]);
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let merged = mlxcel_core::reshape(&output, &[batch, queries, model_dim]);
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.merged", &merged);
        }
        let output = self.output.forward(&merged);
        #[cfg(test)]
        if _layer_index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.output", &output);
        }
        output
    }
}

/// Run the pinned upstream pure-MLX bilinear graph and restore the feature dtype.
pub fn grid_sample_f32(input: &MlxArray, grid: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(input);
    let input_f32 = mlxcel_core::astype(input, mlxcel_core::dtype::FLOAT32);
    let grid_f32 = mlxcel_core::astype(grid, mlxcel_core::dtype::FLOAT32);
    let output = grid_sample_upstream_f32(&input_f32, &grid_f32);
    mlxcel_core::astype(&output, dtype)
}

/// Match the pinned upstream pure-MLX grid-sample graph used outside Metal.
fn grid_sample_upstream_f32(input: &MlxArray, grid: &MlxArray) -> UniquePtr<MlxArray> {
    let input_shape = mlxcel_core::array_shape(input);
    let grid_shape = mlxcel_core::array_shape(grid);
    let (batch, height, width, channels) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (grid_height, grid_width) = (grid_shape[1], grid_shape[2]);
    let f32_scalar = |value| mlxcel_core::full_f32(&[1], value, mlxcel_core::dtype::FLOAT32);
    let i32_scalar = |value| mlxcel_core::from_slice_i32(&[value], &[1]);

    let component = |index| {
        let component = mlxcel_core::slice(
            grid,
            &[0, 0, 0, index],
            &[batch, grid_height, grid_width, index + 1],
        );
        mlxcel_core::reshape(&component, &[batch, grid_height, grid_width])
    };
    let pixel_coordinate = |component: &MlxArray, extent: i32| {
        let coordinate = mlxcel_core::add(component, &f32_scalar(1.0));
        let coordinate = mlxcel_core::multiply_scalar(&coordinate, extent as f32);
        let coordinate = mlxcel_core::subtract(&coordinate, &f32_scalar(1.0));
        mlxcel_core::divide_scalar(&coordinate, 2.0)
    };
    let x = pixel_coordinate(&component(0), width);
    let y = pixel_coordinate(&component(1), height);
    let x0 = mlxcel_core::astype(&mlxcel_core::floor(&x), mlxcel_core::dtype::INT32);
    let y0 = mlxcel_core::astype(&mlxcel_core::floor(&y), mlxcel_core::dtype::INT32);
    let x1 = mlxcel_core::add(&x0, &i32_scalar(1));
    let y1 = mlxcel_core::add(&y0, &i32_scalar(1));

    let to_f32 = |value: &MlxArray| mlxcel_core::astype(value, mlxcel_core::dtype::FLOAT32);
    let expand_weight = |first: &MlxArray, second: &MlxArray| {
        mlxcel_core::expand_dims(&mlxcel_core::multiply(first, second), -1)
    };
    let weight_a = expand_weight(
        &mlxcel_core::subtract(&to_f32(&x1), &x),
        &mlxcel_core::subtract(&to_f32(&y1), &y),
    );
    let weight_b = expand_weight(
        &mlxcel_core::subtract(&x, &to_f32(&x0)),
        &mlxcel_core::subtract(&to_f32(&y1), &y),
    );
    let weight_c = expand_weight(
        &mlxcel_core::subtract(&to_f32(&x1), &x),
        &mlxcel_core::subtract(&y, &to_f32(&y0)),
    );
    let weight_d = expand_weight(
        &mlxcel_core::subtract(&x, &to_f32(&x0)),
        &mlxcel_core::subtract(&y, &to_f32(&y0)),
    );

    let input_flat = mlxcel_core::reshape(input, &[batch, height * width, channels]);
    let gather = |row: &MlxArray, column: &MlxArray| {
        let zero = i32_scalar(0);
        let mut valid = mlxcel_core::logical_and(
            &mlxcel_core::greater_equal(row, &zero),
            &mlxcel_core::less(row, &i32_scalar(height)),
        );
        valid = mlxcel_core::logical_and(&valid, &mlxcel_core::greater_equal(column, &zero));
        valid = mlxcel_core::logical_and(&valid, &mlxcel_core::less(column, &i32_scalar(width)));
        let row = mlxcel_core::clip(row, &zero, &i32_scalar(height - 1));
        let column = mlxcel_core::clip(column, &zero, &i32_scalar(width - 1));
        let index = mlxcel_core::add(&mlxcel_core::multiply(&row, &i32_scalar(width)), &column);
        let index = mlxcel_core::reshape(&index, &[batch, grid_height * grid_width]);
        let index = mlxcel_core::expand_dims(&index, -1);
        let index = mlxcel_core::broadcast_to(&index, &[batch, grid_height * grid_width, channels]);
        let values = mlxcel_core::take_along_axis(&input_flat, &index, 1);
        let values = mlxcel_core::reshape(&values, &[batch, grid_height, grid_width, channels]);
        mlxcel_core::multiply(&values, &mlxcel_core::expand_dims(&valid, -1))
    };

    let term_a = mlxcel_core::multiply(&weight_a, &gather(&y0, &x0));
    let term_b = mlxcel_core::multiply(&weight_b, &gather(&y0, &x1));
    let term_c = mlxcel_core::multiply(&weight_c, &gather(&y1, &x0));
    let term_d = mlxcel_core::multiply(&weight_d, &gather(&y1, &x1));
    let output = mlxcel_core::add(&term_a, &term_b);
    let output = mlxcel_core::add(&output, &term_c);
    mlxcel_core::add(&output, &term_d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_sample_matches_upstream_boundaries_and_sum_order() {
        let input = mlxcel_core::from_slice_f32(&[1.0e10, 1.0, -1.0e10, 1.0], &[1, 2, 2, 1]);
        let grid = mlxcel_core::from_slice_f32(&[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], &[1, 1, 3, 2]);
        let output = grid_sample_f32(&input, &grid);
        assert_eq!(
            mlxcel_core::utils::array_to_vec_f32(&output),
            [2.5e9, 0.25, 0.25]
        );
    }
}
