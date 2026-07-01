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

//! Layout helpers for Qwen3-Next projection and conv split paths.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProjectionLayout {
    pub(super) mixed_qkvz_shape: Vec<i32>,
    pub(super) mixed_ba_shape: Vec<i32>,
    pub(super) q_range: (i32, i32),
    pub(super) k_range: (i32, i32),
    pub(super) v_range: (i32, i32),
    pub(super) z_range: (i32, i32),
    pub(super) b_range: (i32, i32),
    pub(super) a_range: (i32, i32),
    pub(super) v_shape: Vec<i32>,
    pub(super) ba_final_shape: Vec<i32>,
}

pub(super) fn build_projection_layout(
    batch_dims: &[i32],
    num_k_heads: usize,
    head_k_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> ProjectionLayout {
    let nk = num_k_heads as i32;
    let dn = head_k_dim as i32;
    let nv = num_v_heads as i32;
    let dv = head_v_dim as i32;
    let v_size = nv / nk * dv;
    let b_size = nv / nk;

    let mut mixed_qkvz_shape = batch_dims.to_vec();
    mixed_qkvz_shape.push(nk);
    mixed_qkvz_shape.push(-1);

    let mut mixed_ba_shape = batch_dims.to_vec();
    mixed_ba_shape.push(nk);
    mixed_ba_shape.push(-1);

    let v_shape: Vec<i32> = batch_dims.iter().cloned().chain(vec![-1, dv]).collect();
    let ba_final_shape: Vec<i32> = batch_dims.iter().cloned().chain(vec![nv]).collect();

    ProjectionLayout {
        mixed_qkvz_shape,
        mixed_ba_shape,
        q_range: (0, dn),
        k_range: (dn, 2 * dn),
        v_range: (2 * dn, 2 * dn + v_size),
        // z has the same width as v. Do NOT use -1 as the stop: mlxcel_core::slice
        // treats stop = -1 as dim_size - 1 (drops the last element), not "to end",
        // which would silently truncate z (and a) by one column.
        z_range: (2 * dn + v_size, 2 * dn + 2 * v_size),
        b_range: (0, b_size),
        // a has the same width as b; explicit stop for the same slice-semantics reason.
        a_range: (b_size, 2 * b_size),
        v_shape,
        ba_final_shape,
    }
}

pub(super) fn split_conv_output_ranges(key_dim: usize, conv_dim: usize) -> [(i32, i32); 3] {
    let key_dim = key_dim as i32;
    let conv_dim = conv_dim as i32;
    [
        (0, key_dim),
        (key_dim, 2 * key_dim),
        (2 * key_dim, conv_dim),
    ]
}
