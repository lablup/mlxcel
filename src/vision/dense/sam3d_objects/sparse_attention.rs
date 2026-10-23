// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::super::{Array, Grid};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray};

pub fn window_attention(qkv: &MlxArray, grid: &Grid, size: i32, shift: i32) -> Result<Array> {
    let s = mx::array_shape(qkv);
    ensure!(
        s.len() == 4 && s[0] == grid.len() as i32 && s[1] == 3 && s[2] > 0 && s[3] > 0,
        "sparse attention expects [N,3,H,D]"
    );
    let dtype = mx::array_dtype(qkv);
    let n = s[0];
    let heads = s[2];
    let dim = s[3];
    let padded = mx::concatenate(qkv, &mx::zeros(&[1, 3, heads, dim], dtype), 0);
    let mut result = mx::zeros(&[n + 1, heads * dim], dtype);
    for (width, windows) in grid.windows(size, shift)?.iter() {
        let width = *width;
        // Bound each gathered QKV batch to 64 MiB and fallback attention logits
        // to 128 MiB. Padding is masked exactly as in the pinned source.
        let bytes = width
            .checked_mul(3 * heads as usize * dim as usize * mx::array_itemsize(qkv))
            .ok_or_else(|| anyhow::anyhow!("attention gather size overflow"))?;
        let logits = width
            .checked_mul(width)
            .and_then(|v| v.checked_mul(heads as usize * 4))
            .ok_or_else(|| anyhow::anyhow!("attention logits size overflow"))?;
        ensure!(
            bytes <= 64 << 20 && logits <= 128 << 20,
            "single sparse attention window exceeds memory budget"
        );
        let batch = ((64 << 20) / bytes).min((128 << 20) / logits).max(1);
        for windows in windows.chunks(batch) {
            let b = windows.len() as i32;
            let w = width as i32;
            let mut indices = vec![n; windows.len() * width];
            let mut valid = vec![0; indices.len()];
            for (i, rows) in windows.iter().enumerate() {
                indices[i * width..i * width + rows.len()].copy_from_slice(rows);
                valid[i * width..i * width + rows.len()].fill(1);
            }
            let indices = mx::from_slice_i32(&indices, &[b, w]);
            let valid = mx::astype(&mx::from_slice_i32(&valid, &[b, w]), mx::dtype::BOOL);
            let values = mx::take(&padded, &indices, 0);
            let split = |i| {
                mx::transpose_axes(
                    &mx::reshape(
                        &mx::slice(&values, &[0, 0, i, 0, 0], &[b, w, i + 1, heads, dim]),
                        &[b, w, heads, dim],
                    ),
                    &[0, 2, 1, 3],
                )
            };
            let q = split(0);
            let k = split(1);
            let v = split(2);
            let mask = mx::reshape(&valid, &[b, 1, 1, w]);
            // SAFETY: mask is a live MLX array for the duration of this call.
            let out = unsafe {
                mx::scaled_dot_product_attention(&q, &k, &v, (dim as f32).sqrt().recip(), &*mask)
            };
            let out = mx::reshape(
                &mx::transpose_axes(&out, &[0, 2, 1, 3]),
                &[b, w, heads * dim],
            );
            let out = mx::where_cond(
                &mx::reshape(&valid, &[b, w, 1]),
                &out,
                &mx::zeros(&[1], dtype),
            );
            result = mx::scatter_add(
                &result,
                &mx::reshape(&indices, &[b * w]),
                &mx::reshape(&out, &[b * w, 1, heads * dim]),
                0,
            );
            mx::eval(&result);
        }
    }
    Ok(mx::slice(&result, &[0, 0], &[n, heads * dim]))
}
