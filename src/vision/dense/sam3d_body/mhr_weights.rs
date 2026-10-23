// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Checked MHR buffer loading and deterministic NumPy add.at accumulation.
use super::super::layers::Array;
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};

pub(crate) fn reserve<T>(count: usize) -> Result<Vec<T>> {
    ensure!(
        count
            .checked_mul(size_of::<T>())
            .is_some_and(|n| n <= isize::MAX as usize),
        "MHR host allocation exceeds addressable size"
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .context("reserve MHR host buffer")?;
    Ok(values)
}
pub(crate) fn f32_weight(w: &WeightMap, key: &str, dims: &[i32]) -> Result<Array> {
    let x = w
        .get(key)
        .with_context(|| format!("missing MHR buffer {key}"))?;
    ensure!(
        mx::array_dtype(x) == mx::dtype::FLOAT32 && mx::array_shape(x) == dims,
        "MHR {key} must be F32 {dims:?}"
    );
    let valid = mx::all_all(&mx::isfinite(x));
    mx::try_eval(&valid).with_context(|| format!("validate MHR buffer {key}"))?;
    ensure!(mx::item_bool(&valid), "nonfinite MHR buffer {key}");
    Ok(mx::copy(x))
}
pub(crate) fn i32_weight(w: &WeightMap, key: &str, dims: &[i32]) -> Result<(Array, Vec<i32>)> {
    let x = w
        .get(key)
        .with_context(|| format!("missing MHR indices {key}"))?;
    ensure!(
        mx::array_dtype(x) == mx::dtype::INT32 && mx::array_shape(x) == dims,
        "MHR {key} must be I32 {dims:?}"
    );
    let raw = mx::try_array_to_raw_bytes(x)?;
    let mut values = reserve::<i32>(raw.len() / 4)?;
    values.extend(
        raw.chunks_exact(4)
            .map(|v| i32::from_ne_bytes([v[0], v[1], v[2], v[3]])),
    );
    Ok((mx::copy(x), values))
}
pub(crate) fn index_range(values: &[i32], upper: i32, name: &str) -> Result<()> {
    ensure!(
        values.iter().all(|&v| v >= 0 && v < upper),
        "MHR {name} index outside 0..{upper}"
    );
    Ok(())
}
pub(super) fn parents_valid(parents: &[i32]) -> Result<()> {
    ensure!(
        !parents.is_empty()
            && parents
                .iter()
                .enumerate()
                .all(|(i, &p)| p == -1 || (p >= 0 && (p as usize) < i)),
        "MHR parents must be roots (-1) or earlier joints"
    );
    Ok(())
}
/// Accumulate in input order in F32, exactly as the pinned `numpy.add.at`.
/// The last dimension consists of independent columns; no parallel reduction.
pub(super) fn ordered_scatter(values: &MlxArray, indices: &[i32], size: usize) -> Result<Array> {
    let s = mx::array_shape(values);
    ensure!(
        s.len() == 3
            && s[0] > 0
            && s[1] as usize == indices.len()
            && s[2] > 0
            && mx::array_dtype(values) == mx::dtype::FLOAT32,
        "MHR scatter expects F32 [B,N,D]"
    );
    let output_size = i32::try_from(size)?;
    index_range(indices, output_size, "scatter")?;
    let (batch, columns) = (s[0] as usize, s[2] as usize);
    let count = batch
        .checked_mul(size)
        .and_then(|n| n.checked_mul(columns))
        .context("MHR scatter size overflow")?;
    let mut output = reserve::<f32>(count)?;
    output.resize(count, 0.);
    let bytes = mx::try_array_to_raw_bytes(values)?;
    for b in 0..batch {
        for d in 0..columns {
            for (i, &index) in indices.iter().enumerate() {
                let input = ((b * indices.len() + i) * columns + d) * 4;
                let v = f32::from_ne_bytes([
                    bytes[input],
                    bytes[input + 1],
                    bytes[input + 2],
                    bytes[input + 3],
                ]);
                output[(b * size + index as usize) * columns + d] += v;
            }
        }
    }
    Ok(mx::from_slice_f32(&output, &[s[0], output_size, s[2]]))
}
