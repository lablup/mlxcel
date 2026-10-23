// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Bounded typed pointmap loading from one stable file handle.
use anyhow::{Context, Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};
use safetensors::{Dtype, SafeTensors};
use std::{io::Read, path::Path};

pub fn load_pointmap(path: &Path) -> Result<UniquePtr<MlxArray>> {
    let limits = crate::current_image_input_limits();
    let file =
        std::fs::File::open(path).with_context(|| format!("open pointmap {}", path.display()))?;
    load_handle(file, limits.max_decode_alloc_bytes)
}
fn load_handle(file: std::fs::File, max_bytes: u64) -> Result<UniquePtr<MlxArray>> {
    let length = file.metadata()?.len();
    ensure!(
        length >= 8 && length <= max_bytes,
        "pointmap file exceeds input allocation limit"
    );
    let bound = usize::try_from(length)?
        .checked_add(1)
        .context("pointmap byte length overflow")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(bound)
        .context("reserve pointmap input")?;
    file.take(bound as u64).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() == length as usize,
        "pointmap file changed length while reading"
    );
    let tensors = SafeTensors::deserialize(&bytes).context("parse pointmap safetensors")?;
    ensure!(tensors.len() == 1, "pointmap file must contain only points");
    let points = tensors
        .tensor("points")
        .context("pointmap file has no points tensor")?;
    let dtype = match points.dtype() {
        Dtype::F32 => mlxcel_core::dtype::FLOAT32,
        Dtype::F16 => mlxcel_core::dtype::FLOAT16,
        Dtype::BF16 => mlxcel_core::dtype::BFLOAT16,
        _ => anyhow::bail!("pointmap points must be floating point"),
    };
    let shape = points.shape();
    ensure!(
        shape.len() == 3
            && shape[2] == 3
            && shape[0] > 0
            && shape[1] > 0
            && shape.iter().all(|v| *v <= i32::MAX as usize),
        "pointmap must be positive HWC XYZ"
    );
    ensure!(
        shape
            .iter()
            .try_fold(4u64, |n, v| n.checked_mul(*v as u64))
            .is_some_and(|n| n <= max_bytes),
        "pointmap exceeds F32 expansion limit"
    );
    let shape = shape.iter().map(|v| *v as i32).collect::<Vec<_>>();
    Ok(mlxcel_core::from_bytes(points.data(), &shape, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3d_objects_pointmap_reads_validated_handle_after_path_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.safetensors");
        let data = [1.0f32, 2.0, f32::NAN]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![1, 1, 3], &data).unwrap();
        safetensors::serialize_to_file([("points", view)], None, &path).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        std::fs::rename(&path, dir.path().join("original")).unwrap();
        std::fs::write(&path, b"replaced path").unwrap();
        let points = load_handle(file, 4096).unwrap();
        assert_eq!(mlxcel_core::array_shape(&points), [1, 1, 3]);
        let values = mlxcel_core::utils::array_to_vec_f32(&points);
        assert_eq!(&values[..2], &[1.0, 2.0]);
        assert!(values[2].is_nan());
    }
}
