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

//! Relative-position multi-head attention and its helpers.
//!
//! Port of `RelPositionMultiHeadAttention` / `RelPositionalEncoding` from
//! `mlx_audio/stt/models/nemotron_asr/attention.py` and
//! `create_chunked_limited_mask` from `conformer.py` (NeMo's Transformer-XL
//! style attention with untied per-layer `pos_bias_u` / `pos_bias_v`, an
//! additive mask and optional projection biases).

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::layers::copy_weight;

/// Additive fill for blocked attention positions (the reference `NEG_INF`).
pub const NEG_INF: f32 = -1e30;

/// Sinusoidal relative positional embedding `[1, 2L - 1, d_model]` (f32) for
/// a length-`L` input, rows ordered from position `L - 1` down to `-(L - 1)`.
///
/// Equal to the centered slice of the reference's `pe` table (which spans
/// `max_len - 1 .. -(max_len - 1)`) for any `L`, without materializing it.
pub fn rel_pos_embedding(length: usize, d_model: usize) -> UniquePtr<MlxArray> {
    let rows = 2 * length.max(1) - 1;
    let start = length.max(1) as i64 - 1;
    // Same f32 arithmetic as the reference: div_term = exp(2k * -(ln(1e4)/d)).
    let factor = (-(10_000f64.ln() / d_model as f64)) as f32;
    let div_term: Vec<f32> = (0..d_model / 2)
        .map(|k| ((2 * k) as f32 * factor).exp())
        .collect();
    let mut pe = vec![0.0f32; rows * d_model];
    for (row, out) in pe.chunks_exact_mut(d_model).enumerate() {
        let pos = (start - row as i64) as f32;
        for (k, &div) in div_term.iter().enumerate() {
            let angle = pos * div;
            out[2 * k] = angle.sin();
            out[2 * k + 1] = angle.cos();
        }
    }
    mlxcel_core::from_slice_f32(&pe, &[1, rows as i32, d_model as i32])
}

/// Whether query frame `i` may attend to key frame `j` under NeMo's
/// `chunked_limited` style: frames are grouped into chunks of `right + 1`
/// frames and a frame sees its own chunk plus `left / (right + 1)` previous
/// chunks. A negative `left` means unlimited history.
pub fn chunked_limited_visible(i: usize, j: usize, left: i64, right: i64) -> bool {
    let chunk = (right.max(0) + 1) as usize;
    let left_chunks = if left >= 0 {
        left as usize / chunk
    } else {
        usize::MAX
    };
    let (ci, cj) = (i / chunk, j / chunk);
    ci >= cj && ci - cj <= left_chunks
}

/// Additive `[1, 1, T, T]` f32 mask: `0` where visible, [`NEG_INF`] elsewhere.
pub fn chunked_limited_mask(seq_len: usize, left: i64, right: i64) -> UniquePtr<MlxArray> {
    let chunk = right.max(0) + 1;
    let left_chunks = if left >= 0 { left / chunk } else { 100_000_000 };
    let t = seq_len as i32;
    let idx: Vec<i32> = (0..seq_len).map(|i| (i as i64 / chunk) as i32).collect();
    let chunk_idx = mlxcel_core::from_slice_i32(&idx, &[t]);
    let rows = mlxcel_core::reshape(&chunk_idx, &[t, 1]);
    let cols = mlxcel_core::reshape(&chunk_idx, &[1, t]);
    let diff = mlxcel_core::subtract(&rows, &cols);
    let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
    let limit = mlxcel_core::from_slice_i32(&[left_chunks as i32], &[1]);
    let visible = mlxcel_core::logical_and(
        &mlxcel_core::greater_equal(&diff, &zero),
        &mlxcel_core::less_equal(&diff, &limit),
    );
    let open = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    let blocked = mlxcel_core::full_f32(&[1], NEG_INF, mlxcel_core::dtype::FLOAT32);
    let mask = mlxcel_core::where_cond(&visible, &open, &blocked);
    mlxcel_core::reshape(&mask, &[1, 1, t, t])
}

/// Transformer-XL relative shift of `[B, H, Tq, P]` scores: left-pad the last
/// axis by one, view as `[B, H, P + 1, Tq]`, drop the first row, and view
/// back as `[B, H, Tq, P]`.
pub fn rel_shift(x: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let (b, h, tq, p) = (s[0], s[1], s[2], s[3]);
    let padded = mlxcel_core::pad(x, &[0, 0, 0, 0, 0, 0, 1, 0], 0.0);
    let viewed = mlxcel_core::reshape(&padded, &[b, h, p + 1, tq]);
    let dropped = mlxcel_core::slice(&viewed, &[0, 0, 1, 0], &[b, h, p + 1, tq]);
    mlxcel_core::reshape(&dropped, &[b, h, tq, p])
}

pub struct RelPositionMultiHeadAttention {
    linear_q: UnifiedLinear,
    linear_k: UnifiedLinear,
    linear_v: UnifiedLinear,
    linear_out: UnifiedLinear,
    linear_pos: UnifiedLinear,
    pos_bias_u: UniquePtr<MlxArray>,
    pos_bias_v: UniquePtr<MlxArray>,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl RelPositionMultiHeadAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        n_heads: usize,
        d_model: usize,
    ) -> Result<Self, String> {
        let linear =
            |name: &str| UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), 64, 4);
        let head_dim = (d_model / n_heads) as i32;
        let n_heads = n_heads as i32;
        let bias = |name: &str| -> Result<UniquePtr<MlxArray>, String> {
            let b = copy_weight(weights, &format!("{prefix}.{name}"))?;
            Ok(mlxcel_core::reshape(&b, &[1, n_heads, 1, head_dim]))
        };
        Ok(Self {
            linear_q: linear("linear_q")?,
            linear_k: linear("linear_k")?,
            linear_v: linear("linear_v")?,
            linear_out: linear("linear_out")?,
            linear_pos: linear("linear_pos")?,
            pos_bias_u: bias("pos_bias_u")?,
            pos_bias_v: bias("pos_bias_v")?,
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x: [B, T, d]`, `pos_emb: [1, 2T - 1, d]`, `mask`: additive and
    /// broadcastable to `[B, H, T, T]`.
    pub fn forward(
        &self,
        x: &MlxArray,
        pos_emb: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, seq) = (shape[0], shape[1]);
        let heads = |t: &MlxArray, len: i32, b: i32| {
            let r = mlxcel_core::reshape(t, &[b, len, self.n_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&r, &[0, 2, 1, 3])
        };
        let q = heads(&self.linear_q.forward(x), seq, batch);
        let k = heads(&self.linear_k.forward(x), seq, batch);
        let v = heads(&self.linear_v.forward(x), seq, batch);
        let p = self.linear_pos.forward(pos_emb);
        let pos_len = mlxcel_core::array_shape(&p)[1];
        let p = heads(&p, pos_len, 1);

        let q_u = mlxcel_core::add(&q, &self.pos_bias_u);
        let q_v = mlxcel_core::add(&q, &self.pos_bias_v);

        // Position term: rel-shifted, cut to the key length, then scaled.
        let bd = mlxcel_core::matmul(&q_v, &mlxcel_core::swap_axes(&p, -2, -1));
        let bd = rel_shift(&bd);
        let bd_shape = mlxcel_core::array_shape(&bd);
        let bd = mlxcel_core::slice(
            &bd,
            &[0, 0, 0, 0],
            &[bd_shape[0], bd_shape[1], bd_shape[2], seq],
        );
        let bd = mlxcel_core::multiply_scalar(&bd, self.scale);
        let bd = match mask {
            Some(m) => mlxcel_core::add(&bd, m),
            None => bd,
        };
        let bd_ptr: *const MlxArray = &*bd;
        // SAFETY: `bd_ptr` points at `bd`, which outlives the call.
        let o = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q_u, &k, &v, self.scale, bd_ptr)
        };
        let o = mlxcel_core::transpose_axes(&o, &[0, 2, 1, 3]);
        let o = mlxcel_core::reshape(&o, &[batch, seq, self.n_heads * self.head_dim]);
        self.linear_out.forward(&o)
    }
}

impl RelPositionMultiHeadAttention {
    /// Cache-aware step (`RelPositionMultiHeadAttention.stream`): the `c` new
    /// frames `q_in: [B, c, d]` attend to the whole key window
    /// `kv_in: [B, L, d]` (cache followed by the new frames) without a mask,
    /// because the window is exactly the allowed left context.
    /// `pos_emb: [1, 2L - 1, d]` is [`rel_pos_embedding`]`(L)`, the reference's
    /// `RelPositionalEncoding.pos_emb_for(L)`.
    pub fn stream(
        &self,
        q_in: &MlxArray,
        kv_in: &MlxArray,
        pos_emb: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let q_shape = mlxcel_core::array_shape(q_in);
        let kv_shape = mlxcel_core::array_shape(kv_in);
        if q_shape.len() != 3 || kv_shape.len() != 3 || kv_shape[1] < q_shape[1] {
            return Err(format!(
                "streaming attention expects q [B, c, d] and kv [B, L >= c, d], got {q_shape:?} / {kv_shape:?}"
            ));
        }
        let (batch, c, ksz) = (q_shape[0], q_shape[1], kv_shape[1]);
        let heads = |t: &MlxArray, len: i32, b: i32| {
            let r = mlxcel_core::reshape(t, &[b, len, self.n_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&r, &[0, 2, 1, 3])
        };
        let q = heads(&self.linear_q.forward(q_in), c, batch);
        let k = heads(&self.linear_k.forward(kv_in), ksz, batch);
        let v = heads(&self.linear_v.forward(kv_in), ksz, batch);
        let p = self.linear_pos.forward(pos_emb);
        let pos_len = mlxcel_core::array_shape(&p)[1];
        if pos_len != 2 * ksz - 1 {
            return Err(format!(
                "streaming attention pos_emb has {pos_len} rows, expected {}",
                2 * ksz - 1
            ));
        }
        let p = heads(&p, pos_len, 1);

        let q_u = mlxcel_core::add(&q, &self.pos_bias_u);
        let q_v = mlxcel_core::add(&q, &self.pos_bias_v);
        let bd = mlxcel_core::matmul(&q_v, &mlxcel_core::swap_axes(&p, -2, -1));
        let bd = rel_shift(&bd);
        let bd_shape = mlxcel_core::array_shape(&bd);
        let bd = mlxcel_core::slice(
            &bd,
            &[0, 0, 0, 0],
            &[bd_shape[0], bd_shape[1], bd_shape[2], ksz],
        );
        let bd = mlxcel_core::multiply_scalar(&bd, self.scale);
        let bd_ptr: *const MlxArray = &*bd;
        // SAFETY: `bd_ptr` points at `bd`, which outlives the call.
        let o = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q_u, &k, &v, self.scale, bd_ptr)
        };
        let o = mlxcel_core::transpose_axes(&o, &[0, 2, 1, 3]);
        let o = mlxcel_core::reshape(&o, &[batch, c, self.n_heads * self.head_dim]);
        Ok(self.linear_out.forward(&o))
    }
}
