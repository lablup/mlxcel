// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use mlxcel_core::weights::WeightMap;
fn values(a: &MlxArray) -> Vec<f32> {
    mx::utils::array_to_vec_f32(a)
}
fn grid(coords: &[[i32; 4]], r: i32) -> Result<Grid> {
    let flat: Vec<_> = coords.iter().flatten().copied().collect();
    Grid::new(mx::from_slice_i32(&flat, &[coords.len() as i32, 4]), r)
}
#[test]
fn sam3d_objects_sparse_grid_preserves_unsorted_rows_and_subdivision() -> Result<()> {
    let g = grid(&[[0, 1, 0, 0], [0, 0, 0, 0]], 4)?;
    let neighbor = host_i32(g.neighbors()?);
    assert_eq!(neighbor[13], 0);
    assert_eq!(neighbor[4], 1);
    assert_eq!(neighbor[27 + 13], 1);
    assert_eq!(neighbor[27 + 22], 0);
    let (child, parent) = g.downsample()?;
    assert_eq!(child.host_coords(), &[[0, 0, 0, 0]]);
    assert_eq!(host_i32(&parent), [0, 0]);
    let children = g.subdivide()?;
    assert_eq!(children.host_coords()[0], [0, 2, 0, 0]);
    assert_eq!(children.host_coords()[7], [0, 3, 1, 1]);
    assert_eq!(children.host_coords()[8], [0, 0, 0, 0]);
    assert!(grid(&[[0, 0, 0, 0], [0, 0, 0, 0]], 4).is_err());
    assert!(grid(&[[0, 4, 0, 0]], 4).is_err());
    Ok(())
}
#[test]
fn sam3d_objects_sparse_pool_includes_reference_initial_zero() -> Result<()> {
    let g = grid(&[[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], 4)?;
    let x = mx::from_slice_f32(&[3., 6., 6., 3., 8., 4.], &[3, 2]);
    let (out, child, parent) = pool(&x, &g)?;
    assert_eq!(values(&out), [3., 3., 4., 2.]);
    assert_eq!(host_i32(&parent), [0, 0, 1]);
    assert_eq!(child.host_coords(), &[[0, 0, 0, 0], [0, 1, 0, 0]]);
    Ok(())
}
#[test]
fn sam3d_objects_sparse_convolution_parent_projection_matches_expansion() -> Result<()> {
    let parents = grid(&[[0, 0, 0, 0], [0, 1, 0, 0]], 2)?;
    let children = parents.subdivide()?;
    let indices: Vec<i32> = (0..16).map(|i| i / 8).collect();
    let parent = mx::from_slice_i32(&indices, &[16]);
    let x = mx::from_slice_f32(&[1., 2., 3., 4.], &[2, 2]);
    let mut weights = WeightMap::new();
    let w: Vec<_> = (0..3 * 27 * 2)
        .map(|i| (i % 11) as f32 * 0.01 - 0.05)
        .collect();
    weights.insert(
        "conv.conv.weight".into(),
        mx::from_slice_f32(&w, &[3, 3, 3, 3, 2]),
    );
    weights.insert(
        "conv.conv.bias".into(),
        mx::from_slice_f32(&[0.1, 0.2, 0.3], &[3]),
    );
    let conv = SparseConv::from_weights(&weights, "conv", 2, 3, 3)?;
    let actual = values(&*conv.from_parents(&x, &parent, &children)?);
    let expected = values(&*conv.forward(&mx::take(&x, &parent, 0), &children)?);
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.into_iter().zip(expected) {
        assert!((a - b).abs() < 2e-6, "{a} != {b}");
    }
    Ok(())
}
#[test]
fn sam3d_objects_sparse_attention_masks_padding_and_isolates_batches() -> Result<()> {
    let g = grid(&[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], 2)?;
    let qkv = mx::from_slice_f32(
        &[0., 0., 2., 0., 0., 4., 0., 0., 9., 0., 0., 20.],
        &[4, 3, 1, 1],
    );
    let actual = values(&*window_attention(&qkv, &g, 2, 0)?);
    for (a, b) in actual.into_iter().zip([5., 5., 5., 20.]) {
        assert!((a - b).abs() < 2e-6, "{a} != {b}");
    }
    Ok(())
}
#[test]
fn sam3d_objects_sparse_group_norm_spans_voxels_and_channels() -> Result<()> {
    let mut weights = WeightMap::new();
    weights.insert("norm.weight".into(), mx::ones(&[4], mx::dtype::FLOAT32));
    weights.insert("norm.bias".into(), mx::zeros(&[4], mx::dtype::FLOAT32));
    let norm = SparseGroupNorm::from_weights(&weights, "norm", 4, 2)?;
    let x = mx::from_slice_f32(&[1., 3., 10., 12., 5., 7., 14., 16.], &[2, 4]);
    let actual = values(&*norm.forward(&x)?);
    let scale = (5.0f32 + 1e-5).sqrt();
    for (a, b) in actual.into_iter().zip([-3., -1., -3., -1., 1., 3., 1., 3.]) {
        assert!((a - b / scale).abs() < 2e-6);
    }
    Ok(())
}

#[test]
fn sam3d_objects_sparse_windows_are_cached_and_keep_stable_ties() -> Result<()> {
    let g = grid(&[[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 3, 0, 0]], 4)?;
    let before = g.resident_bytes()?;
    let a = g.windows(2, 0)?;
    let b = g.windows(2, 0)?;
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert!(g.resident_bytes()? > before);
    assert_eq!(
        a.as_ref(),
        &vec![(1, vec![vec![3]]), (4, vec![vec![0, 1, 2]])]
    );
    let clone = g.clone();
    assert!(std::sync::Arc::ptr_eq(&a, &clone.windows(2, 0)?));
    assert!(!std::sync::Arc::ptr_eq(&a, &g.windows(2, 1)?));
    assert!(g.windows(0, 0).is_err());
    let too_large = grid(&[[0, 0, 0, 0]], 1_048_576)?;
    assert!(too_large.subdivide().is_err());
    Ok(())
}
