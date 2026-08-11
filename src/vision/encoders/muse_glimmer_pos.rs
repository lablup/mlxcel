// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Muse Glimmer learned-position interpolation and vision RoPE helpers.

use mlxcel_core::{MlxArray, UniquePtr};

pub fn bilinear_indices_and_weights(
    grid_thw: &[(i32, i32, i32)],
    table_h: usize,
    table_w: usize,
) -> Result<(Vec<i32>, Vec<f32>), String> {
    if table_h == 0 || table_w == 0 {
        return Err("Muse Glimmer position table side must be non-zero".to_string());
    }
    let mut idx_parts = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut weight_parts = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for &(t, h, w) in grid_thw {
        if t <= 0 || h <= 0 || w <= 0 {
            return Err(format!(
                "Muse Glimmer grid must be positive, got {:?}",
                (t, h, w)
            ));
        }
        for _ in 0..t {
            for y in 0..h {
                let (y0, y1, yw0, yw1) = axis_bilinear(y, h, table_h);
                for x in 0..w {
                    let (x0, x1, xw0, xw1) = axis_bilinear(x, w, table_w);
                    idx_parts[0].push((y0 * table_w + x0) as i32);
                    idx_parts[1].push((y0 * table_w + x1) as i32);
                    idx_parts[2].push((y1 * table_w + x0) as i32);
                    idx_parts[3].push((y1 * table_w + x1) as i32);
                    weight_parts[0].push(yw0 * xw0);
                    weight_parts[1].push(yw0 * xw1);
                    weight_parts[2].push(yw1 * xw0);
                    weight_parts[3].push(yw1 * xw1);
                }
            }
        }
    }
    let token_count: usize = idx_parts.iter().map(Vec::len).sum::<usize>() / 4;
    let mut indices = Vec::with_capacity(token_count * 4);
    let mut weights = Vec::with_capacity(token_count * 4);
    for part in idx_parts {
        indices.extend(part);
    }
    for part in weight_parts {
        weights.extend(part);
    }
    Ok((indices, weights))
}

pub fn interpolate_position_table(
    table: &MlxArray,
    grid_thw: &[(i32, i32, i32)],
    table_h: usize,
    table_w: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(table);
    if shape.len() != 2 || shape[0] as usize != table_h * table_w {
        return Err(format!(
            "Muse Glimmer position table shape must be [{}, hidden], got {shape:?}",
            table_h * table_w
        ));
    }
    let (indices, weights) = bilinear_indices_and_weights(grid_thw, table_h, table_w)?;
    let tokens = (indices.len() / 4) as i32;
    let hidden = shape[1];
    let idx = mlxcel_core::from_slice_i32(&indices, &[tokens * 4]);
    let gathered = mlxcel_core::take(table, &idx, 0);
    let gathered = mlxcel_core::reshape(&gathered, &[4, tokens, hidden]);
    let w = mlxcel_core::from_slice_f32(&weights, &[4, tokens, 1]);
    let weighted = mlxcel_core::multiply(&gathered, &w);
    Ok(mlxcel_core::sum_axis(&weighted, 0, false))
}

pub fn muse_2d_rope(
    grid_thw: &[(i32, i32, i32)],
    head_dim: usize,
    theta: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    if !head_dim.is_multiple_of(4) {
        return Err(format!(
            "Muse Glimmer vision head_dim must be divisible by 4 for width/height interleaved RoPE, got {head_dim}"
        ));
    }
    let spatial_dim = head_dim / 2;
    let axis_freqs = spatial_dim / 2;
    let mut inv_freq = Vec::with_capacity(axis_freqs);
    for i in 0..axis_freqs {
        inv_freq.push(1.0 / theta.powf((2 * i) as f32 / spatial_dim as f32));
    }

    let total_tokens: usize = grid_thw
        .iter()
        .map(|&(t, h, w)| {
            if t <= 0 || h <= 0 || w <= 0 {
                Err(format!(
                    "Muse Glimmer grid must be positive, got {:?}",
                    (t, h, w)
                ))
            } else {
                Ok((t * h * w) as usize)
            }
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .sum();
    let mut values = Vec::with_capacity(total_tokens * head_dim);
    for &(t, h, w) in grid_thw {
        for _ in 0..t {
            for y in 0..h {
                for x in 0..w {
                    let w_freq = inv_freq.iter().map(|f| (x + 1) as f32 * *f);
                    let h_freq = inv_freq.iter().map(|f| (y + 1) as f32 * *f);
                    values.extend(w_freq.clone());
                    values.extend(h_freq.clone());
                    values.extend(w_freq);
                    values.extend(h_freq);
                }
            }
        }
    }
    Ok(mlxcel_core::from_slice_f32(
        &values,
        &[total_tokens as i32, head_dim as i32],
    ))
}

fn axis_bilinear(pos: i32, grid: i32, table: usize) -> (usize, usize, f32, f32) {
    let src = (pos as f32 + 0.5) * table as f32 / grid as f32 - 0.5;
    let lo_f = src.floor();
    let hi_f = lo_f + 1.0;
    let hi_weight = src - lo_f;
    let lo_weight = 1.0 - hi_weight;
    (
        clamp_index(lo_f as i32, table),
        clamp_index(hi_f as i32, table),
        if lo_f < 0.0 || lo_f >= table as f32 {
            0.0
        } else {
            lo_weight
        },
        if hi_f < 0.0 || hi_f >= table as f32 {
            0.0
        } else {
            hi_weight
        },
    )
}

fn clamp_index(idx: i32, table: usize) -> usize {
    idx.clamp(0, table as i32 - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
        let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&f);
        mlxcel_core::array_to_raw_bytes(&f)
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn bilinear_indices_match_32_by_32_corner_planes() {
        let (indices, weights) = bilinear_indices_and_weights(&[(1, 16, 16)], 32, 32).unwrap();
        let n = 16 * 16;

        assert_eq!(indices.len(), 4 * n);
        assert_eq!(weights.len(), 4 * n);
        assert_eq!(
            [indices[0], indices[n], indices[2 * n], indices[3 * n]],
            [0, 1, 32, 33]
        );
        for offset in [0, n, 2 * n, 3 * n] {
            assert_close(weights[offset], 0.25);
        }
    }

    #[test]
    fn interpolation_uses_zero_padding_at_runtime_grid_edges() {
        let mut table = vec![0.0f32; 32 * 32];
        table[0] = 1.0;
        let table = mlxcel_core::from_slice_f32(&table, &[32 * 32, 1]);
        let out = interpolate_position_table(&table, &[(1, 64, 64)], 32, 32).unwrap();
        let values = to_vec_f32(&out);

        assert_eq!(mlxcel_core::array_shape(&out), vec![64 * 64, 1]);
        assert_close(values[0], 0.5625);
    }

    #[test]
    fn interpolation_samples_32_by_32_center_to_single_runtime_patch() {
        let table = (0..32)
            .flat_map(|y| (0..32).map(move |x| y as f32 * 100.0 + x as f32))
            .collect::<Vec<_>>();
        let table = mlxcel_core::from_slice_f32(&table, &[32 * 32, 1]);
        let out = interpolate_position_table(&table, &[(1, 1, 1)], 32, 32).unwrap();
        let values = to_vec_f32(&out);

        assert_eq!(mlxcel_core::array_shape(&out), vec![1, 1]);
        assert_close(values[0], 1565.5);
    }

    #[test]
    fn rope_uses_width_height_interleaved_reference_order() {
        let rope = muse_2d_rope(&[(1, 2, 3)], 8, 10_000.0).unwrap();
        let values = to_vec_f32(&rope);
        let second_token = &values[8..16];

        assert_eq!(mlxcel_core::array_shape(&rope), vec![6, 8]);
        let expected = [2.0, 0.02, 1.0, 0.01, 2.0, 0.02, 1.0, 0.01];
        for (actual, expected) in second_token.iter().zip(expected) {
            assert_close(*actual, expected);
        }

        let concatenated_axes = [2.0, 0.02, 2.0, 0.02, 1.0, 0.01, 1.0, 0.01];
        assert_ne!(second_token, concatenated_axes);
    }

    #[test]
    fn rope_plus_one_offset_changes_the_origin_token() {
        let rope = muse_2d_rope(&[(1, 1, 1)], 8, 10_000.0).unwrap();
        let values = to_vec_f32(&rope);
        let expected = [1.0, 0.01, 1.0, 0.01, 1.0, 0.01, 1.0, 0.01];

        for (actual, expected) in values.iter().zip(expected) {
            assert_close(*actual, expected);
        }
        assert_ne!(values, vec![0.0; 8]);
    }
}
