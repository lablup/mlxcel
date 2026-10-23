// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn put(w: &mut WeightMap, key: &str, values: &[f32], dims: &[i32], dtype: i32) {
    w.insert(
        key.into(),
        mx::astype(&mx::from_slice_f32(values, dims), dtype),
    );
}

#[test]
#[allow(clippy::excessive_precision)] // Preserve the authenticated oracle's exact literals.
fn sam3d_body_linear_preserves_bf16_and_authentic_addmm_rounding() -> Result<()> {
    // Existing authenticated Sapiens2 MLX oracle (source JSON SHA256
    // ab93240df9362f37bdc00fa5a179cf850630af7a9903c13d2eae52a2098e35b8).
    let mut w = WeightMap::new();
    put(
        &mut w,
        "linear.weight",
        &[
            0.80078125,
            -0.1640625,
            0.267578125,
            -1.3125,
            0.76171875,
            -0.73046875,
            0.97265625,
            -0.57421875,
            1.015625,
            -1.8203125,
            2.765625,
            0.6484375,
        ],
        &[3, 4],
        mx::dtype::BFLOAT16,
    );
    put(
        &mut w,
        "linear.bias",
        &[1.3671875, -0.70703125, 0.82421875],
        &[3],
        mx::dtype::BFLOAT16,
    );
    let layer = Linear::from_weights(&w, "linear")?;
    assert_eq!(mx::array_dtype(&layer.weight), mx::dtype::BFLOAT16);
    let x = mx::astype(
        &mx::from_slice_f32(&[-0.447265625, 0.84375, 1.140625, -0.08935546875], &[1, 4]),
        mx::dtype::BFLOAT16,
    );
    let y = layer.forward(&x);
    assert_eq!(
        mx::utils::array_to_vec_f32(&y),
        [1.296875, -0.50390625, 1.9296875]
    );
    Ok(())
}

#[test]
fn sam3d_body_norm_matches_analytic_affine_and_restores_upcast_dtype() -> Result<()> {
    for dtype in [mx::dtype::FLOAT32, mx::dtype::BFLOAT16] {
        let mut w = WeightMap::new();
        put(&mut w, "norm.weight", &[2., 4.], &[2], dtype);
        put(&mut w, "norm.bias", &[0.25, -0.5], &[2], dtype);
        let norm = Norm::from_weights(&w, "norm", 3., true)?;
        let x = mx::astype(&mx::from_slice_f32(&[-1., 1.], &[1, 2]), dtype);
        let y = norm.forward(&x);
        assert_eq!(mx::array_dtype(&y), dtype);
        assert_eq!(mx::utils::array_to_vec_f32(&y), [-0.75, 1.5]);
    }
    Ok(())
}

#[test]
fn sam3d_body_layers_reject_malformed_weights() {
    let mut w = WeightMap::new();
    assert!(Linear::from_weights(&w, "l").is_err());
    put(&mut w, "l.weight", &[1., 2.], &[2], mx::dtype::FLOAT32);
    assert!(Linear::from_weights(&w, "l").is_err());
    put(&mut w, "l.weight", &[1., 2.], &[1, 2], mx::dtype::FLOAT32);
    put(&mut w, "l.bias", &[1., 2.], &[2], mx::dtype::FLOAT32);
    assert!(Linear::from_weights(&w, "l").is_err());
    put(&mut w, "n.weight", &[1., 1.], &[2], mx::dtype::FLOAT32);
    put(&mut w, "n.bias", &[0.], &[1], mx::dtype::FLOAT32);
    assert!(Norm::from_weights(&w, "n", 1e-5, false).is_err());
    put(&mut w, "n.bias", &[0., 0.], &[2], mx::dtype::FLOAT32);
    assert!(Norm::from_weights(&w, "n", f32::NAN, false).is_err());
    assert!(Norm::from_weights(&w, "n", 0., false).is_err());
}

#[test]
fn sam3d_body_array_preserves_integer_topology_but_linear_rejects_it() -> Result<()> {
    let mut w = WeightMap::new();
    w.insert("faces".into(), mx::from_slice_i32(&[0, 1, 2], &[1, 3]));
    assert_eq!(mx::array_dtype(&*array(&w, "faces")?), mx::dtype::INT32);
    w.insert("linear.weight".into(), mx::from_slice_i32(&[1, 2], &[1, 2]));
    assert!(Linear::from_weights(&w, "linear").is_err());
    Ok(())
}
