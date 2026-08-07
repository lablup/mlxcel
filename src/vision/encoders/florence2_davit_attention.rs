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

//! The two attention flavors of the Florence-2 DaViT backbone.
//!
//! `WindowAttention` is ordinary multi-head self-attention restricted to
//! non-overlapping spatial windows. `ChannelAttention` transposes the
//! problem: channels attend to channels, with the token axis as the
//! reduction dimension. Running both in every block is what makes the
//! backbone "dual attention".
//!
//! Reference: mlx-vlm `mlx_vlm/models/florence2/vision.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py>).

use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::deepseekocr_sam::{window_partition, window_unpartition};

/// Split a packed `(B, N, 3 * heads * head_dim)` qkv projection into three
/// `(B, heads, N, head_dim)` tensors.
fn split_qkv(
    qkv: &MlxArray,
    b: i32,
    n: i32,
    heads: i32,
    head_dim: i32,
) -> [UniquePtr<MlxArray>; 3] {
    let packed = mlxcel_core::reshape(qkv, &[b, n, 3, heads, head_dim]);
    std::array::from_fn(|i| {
        let i = i as i32;
        let part = mlxcel_core::slice(&packed, &[0, 0, i, 0, 0], &[b, n, i + 1, heads, head_dim]);
        let part = mlxcel_core::reshape(&part, &[b, n, heads, head_dim]);
        mlxcel_core::transpose_axes(&part, &[0, 2, 1, 3])
    })
}

/// Windowed multi-head self-attention over non-overlapping `window_size`
/// tiles, with the feature map padded up to a whole number of windows and
/// cropped back afterwards.
pub(crate) struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: i32,
    window_size: i32,
    scale: f32,
}

impl WindowAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dim: i32,
        num_heads: i32,
        window_size: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            qkv: Linear::from_weights(weights, &format!("{prefix}.qkv"))?,
            proj: Linear::from_weights(weights, &format!("{prefix}.proj"))?,
            num_heads,
            window_size,
            scale: ((dim / num_heads) as f32).powf(-0.5),
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, c) = (shape[0], shape[2]);
        let (h, w) = size;
        let ws = self.window_size;

        let grid = mlxcel_core::reshape(x, &[b, h, w, c]);
        let (windows, pad_hw) = window_partition(&grid, ws);
        let win_count = mlxcel_core::array_shape(&windows)[0];
        let n = ws * ws;
        let tokens = mlxcel_core::reshape(&windows, &[win_count, n, c]);

        let head_dim = c / self.num_heads;
        let qkv = self.qkv.forward(&tokens);
        let [q, k, v] = split_qkv(&qkv, win_count, n, self.num_heads, head_dim);

        let q = mlxcel_core::multiply_scalar(&q, self.scale);
        let attn = mlxcel_core::matmul(&q, &mlxcel_core::transpose_axes(&k, &[0, 1, 3, 2]));
        let attn = mlxcel_core::softmax(&attn, -1);
        let out = mlxcel_core::matmul(&attn, &v);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[win_count, n, c]);
        let out = self.proj.forward(&out);

        let out = mlxcel_core::reshape(&out, &[-1, ws, ws, c]);
        let merged = window_unpartition(&out, ws, pad_hw, (h, w));
        mlxcel_core::reshape(&merged, &[b, h * w, c])
    }
}

/// Channel self-attention: the attention matrix is `(C/groups, C/groups)`
/// per group, so channels attend to channels while the token axis is the
/// reduction dimension.
///
/// Two details are easy to get wrong and are deliberate here. First, the
/// query scale is the token count `N`, not the head dimension. Second, the
/// upstream inline comment annotates the attention matrix as
/// `(B, groups, N, N)`; the actual product of `(B, g, C/g, N)` and
/// `(B, g, N, C/g)` is `(B, groups, C/groups, C/groups)`, which is the whole
/// point of channel attention. This port follows the code, not the comment.
pub(crate) struct ChannelAttention {
    qkv: Linear,
    proj: Linear,
    groups: i32,
}

impl ChannelAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        groups: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            qkv: Linear::from_weights(weights, &format!("{prefix}.qkv"))?,
            proj: Linear::from_weights(weights, &format!("{prefix}.proj"))?,
            groups,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let group_dim = c / self.groups;

        let qkv = self.qkv.forward(x);
        let [q, k, v] = split_qkv(&qkv, b, n, self.groups, group_dim);

        // Token-count scale, per the reference.
        let q = mlxcel_core::multiply_scalar(&q, (n as f32).powf(-0.5));
        let attn = mlxcel_core::matmul(&mlxcel_core::transpose_axes(&q, &[0, 1, 3, 2]), &k);
        // The reduction runs over N (up to ~37k tokens at stage 0), so the
        // logits can be large; `softmax_precise` accumulates the exp/sum in
        // f32 internally. Its output dtype is still `at_least_float(attn's
        // own dtype)`, i.e. unchanged here since `attn` is already a float
        // tensor, so there is nothing to cast back afterward.
        let attn = mlxcel_core::softmax_precise(&attn, -1);

        let out = mlxcel_core::matmul(&attn, &mlxcel_core::transpose_axes(&v, &[0, 1, 3, 2]));
        let out = mlxcel_core::transpose_axes(&out, &[0, 1, 3, 2]);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, n, c]);
        self.proj.forward(&out)
    }
}
