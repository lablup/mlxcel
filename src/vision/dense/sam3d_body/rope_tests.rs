// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
fn sam3d_body_rope_uses_split_half_preserves_prefix_and_promotes_tables() -> Result<()> {
    let values: Vec<f32> = (0..28).map(|i| i as f32).collect();
    let q = mx::astype(
        &mx::from_slice_f32(&values, &[1, 1, 7, 4]),
        mx::dtype::BFLOAT16,
    );
    let k = mx::negative(&q);
    let sin = mx::ones(&[2, 4], mx::dtype::FLOAT32);
    let cos = mx::zeros(&[2, 4], mx::dtype::FLOAT32);
    let (q, k) = apply_rope(&q, &k, &sin, &cos, 5)?;
    let mut expected = values[..20].to_vec();
    expected.extend_from_slice(&[-22., -23., 20., 21., -26., -27., 24., 25.]);
    assert_eq!(mx::array_dtype(&q), mx::dtype::FLOAT32);
    assert_eq!(mx::utils::array_to_vec_f32(&q), expected);
    assert_eq!(
        mx::utils::array_to_vec_f32(&k),
        expected.iter().map(|x| -x).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn sam3d_body_rope_axis_order_and_repeated_half() -> Result<()> {
    let mut w = WeightMap::new();
    w.insert(
        "rope.periods".into(),
        mx::astype(&mx::from_slice_f32(&[1.], &[1]), mx::dtype::BFLOAT16),
    );
    let rope = Rope::from_weights(&w, "rope", 4)?;
    let (sin, cos) = rope.forward(1, 1)?;
    assert_eq!(mx::utils::array_to_vec_f32(&sin), [0.; 4]);
    assert_eq!(mx::utils::array_to_vec_f32(&cos), [1.; 4]);
    let (_, cos) = rope.forward(1, 2)?;
    assert_eq!(mx::array_dtype(&cos), mx::dtype::FLOAT32);
    assert_eq!(
        mx::utils::array_to_vec_f32(&cos),
        [1., -1., 1., -1., 1., -1., 1., -1.]
    );
    assert!(rope.forward(0, 1).is_err());
    assert!(rope.forward(usize::MAX, 2).is_err());
    assert!(Rope::from_weights(&w, "rope", 6).is_err());
    assert!(Rope::from_weights(&w, "rope", 8).is_err());
    Ok(())
}

#[test]
fn sam3d_body_rope_rejects_malformed_tables_and_prefix() {
    let q = mx::zeros(&[1, 1, 7, 4], mx::dtype::FLOAT32);
    let table = mx::zeros(&[2, 4], mx::dtype::FLOAT32);
    assert!(apply_rope(&q, &q, &table, &table, 7).is_err());
    assert!(apply_rope(&q, &q, &table, &table, 4).is_err());
    let wrong = mx::zeros(&[2, 3], mx::dtype::FLOAT32);
    assert!(apply_rope(&q, &q, &table, &wrong, 5).is_err());
}
