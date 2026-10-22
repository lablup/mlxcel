// Copyright 2025-2026 Lablup Inc.
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

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub(super) struct SamAttention {
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
}

impl SamAttention {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        num_heads: i32,
        downsample_rate: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let internal_dim = hidden_size / downsample_rate;
        if internal_dim % num_heads != 0 {
            return Err(format!(
                "SAM attention width {internal_dim} is not divisible by {num_heads} heads"
            ));
        }
        let head_dim = internal_dim / num_heads;
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        Ok(Self {
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            o_proj: linear("o_proj")?,
        })
    }

    pub(super) fn forward(&self, q: &MlxArray, k: &MlxArray, v: &MlxArray) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(q);
        let k_shape = mlxcel_core::array_shape(k);
        let (batch, q_len, k_len) = (q_shape[0], q_shape[1], k_shape[1]);
        let q = mlxcel_core::reshape(
            &self.q_proj.forward(q),
            &[batch, q_len, self.num_heads, self.head_dim],
        );
        let k = mlxcel_core::reshape(
            &self.k_proj.forward(k),
            &[batch, k_len, self.num_heads, self.head_dim],
        );
        let v = mlxcel_core::reshape(
            &self.v_proj.forward(v),
            &[batch, k_len, self.num_heads, self.head_dim],
        );
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        // SAFETY: a null mask selects unmasked bidirectional attention; q/k/v
        // remain live for the complete FFI call.
        let out = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        self.o_proj.forward(&mlxcel_core::reshape(
            &out,
            &[batch, q_len, self.num_heads * self.head_dim],
        ))
    }
}

/// Multi-head attention with interleaved 2D rotary frequencies.
pub struct RopeAttention {
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    freqs_cos: UniquePtr<MlxArray>,
    freqs_sin: UniquePtr<MlxArray>,
    repeat_freqs_k: bool,
}

impl RopeAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        num_heads: i32,
        downsample_rate: i32,
        _kv_dim: i32,
        feat_size: (i32, i32),
        theta: f32,
        repeat_freqs_k: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let internal_dim = hidden_size / downsample_rate;
        if internal_dim % num_heads != 0 {
            return Err(format!(
                "SAM3 RoPE attention width {internal_dim} is not divisible by {num_heads} heads"
            ));
        }
        let head_dim = internal_dim / num_heads;
        if head_dim % 4 != 0 {
            return Err(format!(
                "SAM3 RoPE head width must be divisible by 4, got {head_dim}"
            ));
        }
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        let (freqs_cos, freqs_sin) = init_2d_freqs(head_dim, feat_size, theta);
        Ok(Self {
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            o_proj: linear("o_proj")?,
            freqs_cos,
            freqs_sin,
            repeat_freqs_k,
        })
    }

    #[must_use]
    pub fn forward(&self, q: &MlxArray, k: &MlxArray, v: &MlxArray) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(q);
        let k_shape = mlxcel_core::array_shape(k);
        let (batch, q_len, k_len) = (q_shape[0], q_shape[1], k_shape[1]);
        let q = mlxcel_core::reshape(
            &self.q_proj.forward(q),
            &[batch, q_len, self.num_heads, self.head_dim],
        );
        let k = mlxcel_core::reshape(
            &self.k_proj.forward(k),
            &[batch, k_len, self.num_heads, self.head_dim],
        );
        let v = mlxcel_core::reshape(
            &self.v_proj.forward(v),
            &[batch, k_len, self.num_heads, self.head_dim],
        );
        let (q, k) = apply_rope(
            &q,
            &k,
            &self.freqs_cos,
            &self.freqs_sin,
            self.repeat_freqs_k,
        );
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        // SAFETY: a null mask selects unmasked bidirectional attention; q/k/v
        // remain live for the complete FFI call.
        let out = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        self.o_proj.forward(&mlxcel_core::reshape(
            &out,
            &[batch, q_len, self.num_heads * self.head_dim],
        ))
    }
}

fn init_2d_freqs(
    dim: i32,
    feat_size: (i32, i32),
    theta: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let half = dim / 2;
    let indices = mlxcel_core::arange_f32(0.0, half as f32, 2.0);
    let freqs = mlxcel_core::exp(&mlxcel_core::multiply_scalar(
        &indices,
        -theta.ln() / half as f32,
    ));
    let mut xs = Vec::with_capacity((feat_size.0 * feat_size.1) as usize);
    let mut ys = Vec::with_capacity(xs.capacity());
    for y in 0..feat_size.0 {
        for x in 0..feat_size.1 {
            xs.push(x as f32);
            ys.push(y as f32);
        }
    }
    let x = mlxcel_core::from_slice_f32(&xs, &[xs.len() as i32]);
    let y = mlxcel_core::from_slice_f32(&ys, &[ys.len() as i32]);
    let angles = mlxcel_core::concatenate(
        &mlxcel_core::outer(&x, &freqs),
        &mlxcel_core::outer(&y, &freqs),
        -1,
    );
    (mlxcel_core::cos(&angles), mlxcel_core::sin(&angles))
}

fn rope_table(table: &MlxArray, len: i32, repeat: bool) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(table);
    let table = if repeat && len > shape[0] {
        mlxcel_core::tile(table, &[(len + shape[0] - 1) / shape[0], 1])
    } else {
        mlxcel_core::copy(table)
    };
    let table = mlxcel_core::slice(&table, &[0, 0], &[len, shape[1]]);
    mlxcel_core::reshape(&table, &[1, len, 1, shape[1]])
}

fn pair_parts(x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(x);
    let paired = mlxcel_core::reshape(x, &[shape[0], shape[1], shape[2], shape[3] / 2, 2]);
    let real = mlxcel_core::squeeze_axis(
        &mlxcel_core::slice(
            &paired,
            &[0, 0, 0, 0, 0],
            &[shape[0], shape[1], shape[2], shape[3] / 2, 1],
        ),
        -1,
    );
    let imag = mlxcel_core::squeeze_axis(
        &mlxcel_core::slice(
            &paired,
            &[0, 0, 0, 0, 1],
            &[shape[0], shape[1], shape[2], shape[3] / 2, 2],
        ),
        -1,
    );
    (real, imag)
}

fn rotate_one(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let (real, imag) = pair_parts(x);
    let out_real = mlxcel_core::subtract(
        &mlxcel_core::multiply(&real, cos),
        &mlxcel_core::multiply(&imag, sin),
    );
    let out_imag = mlxcel_core::add(
        &mlxcel_core::multiply(&real, sin),
        &mlxcel_core::multiply(&imag, cos),
    );
    mlxcel_core::reshape(&mlxcel_core::stack_owned(&[out_real, out_imag], -1), &shape)
}

fn apply_rope(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
    repeat_k: bool,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let q_len = mlxcel_core::array_shape(q)[1];
    let k_len = mlxcel_core::array_shape(k)[1];
    let cos_q = rope_table(cos, q_len, false);
    let sin_q = rope_table(sin, q_len, false);
    let cos_k = rope_table(cos, k_len, repeat_k);
    let sin_k = rope_table(sin, k_len, repeat_k);
    (rotate_one(q, &cos_q, &sin_q), rotate_one(k, &cos_k, &sin_k))
}
