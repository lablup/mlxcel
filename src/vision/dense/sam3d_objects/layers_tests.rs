// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Numerical contracts that distinguish guidance bias and norm semantics.
use super::*;

#[test]
fn unconditional_cross_attention_retains_value_and_output_bias() -> Result<()> {
    let mut weights = WeightMap::new();
    for (name, values, shape) in [
        ("a.to_q.weight", vec![1., 0., 0., 1.], vec![2, 2]),
        ("a.to_kv.weight", vec![0.; 8], vec![4, 2]),
        ("a.to_kv.bias", vec![19., -12., 2., -3.], vec![4]),
        ("a.to_out.weight", vec![2., 1., -1., 3.], vec![2, 2]),
        ("a.to_out.bias", vec![5., 7.], vec![2]),
    ] {
        weights.insert(name.into(), mx::from_slice_f32(&values, &shape));
    }
    let attention = Attention::from_weights(&weights, "a", 1, true, false)?;
    let condition = attention.prepare(None)?;
    let x = mx::zeros(&[1, 3, 2], mx::dtype::FLOAT32);
    let actual = attention.forward(&x, Some(&condition))?;
    assert_eq!(mx::array_shape(&actual), [1, 3, 2]);
    assert_eq!(
        mx::utils::array_to_vec_f32(&actual),
        [6., -4., 6., -4., 6., -4.]
    );
    Ok(())
}

#[test]
fn head_norm_keeps_bf16_and_finite_zero_heads() -> Result<()> {
    let x = mx::astype(
        &mx::from_slice_f32(&[0., 0., 3., 4.], &[1, 1, 2, 2]),
        mx::dtype::BFLOAT16,
    );
    let gamma = mx::full_f32(&[2, 2], 1., mx::dtype::BFLOAT16);
    let actual = mx::compiled_objects_head_norm(&x, &gamma)?;
    assert_eq!(mx::array_dtype(&actual), mx::dtype::BFLOAT16);
    let values = mx::utils::array_to_vec_f32(&actual);
    assert_eq!(&values[..2], &[0., 0.]);
    assert!((values[2] - 0.6 * 2_f32.sqrt()).abs() < 0.004);
    assert!((values[3] - 0.8 * 2_f32.sqrt()).abs() < 0.004);
    Ok(())
}

#[test]
fn compiled_layers_reject_invalid_shapes_and_dtypes() {
    let x = mx::zeros(&[1, 2, 4], mx::dtype::BFLOAT16);
    let wrong_width = mx::zeros(&[1, 1, 3], mx::dtype::BFLOAT16);
    let wrong_dtype = mx::zeros(&[1, 1, 4], mx::dtype::FLOAT32);
    assert!(mx::compiled_objects_modulate(&x, &wrong_width, &x).is_err());
    assert!(mx::compiled_objects_modulate(&x, &wrong_dtype, &x).is_err());
    assert!(mx::compiled_objects_gated_residual(&x, &wrong_width, &x).is_err());
    assert!(mx::compiled_objects_head_norm(&x, &wrong_width).is_err());
}

#[test]
fn position_embedding_preserves_axis_order_and_zero_padding() {
    let coords = mx::from_slice_i32(&[0, 0, 0], &[1, 3]);
    let actual = position_embedding(&coords, 14);
    assert_eq!(mx::array_shape(&actual), [1, 14]);
    assert_eq!(
        mx::utils::array_to_vec_f32(&actual),
        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.]
    );
}

fn layer_oracle() -> Result<WeightMap> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sam3d_objects/layers.safetensors");
    mx::weights::load_safetensors(path).map_err(anyhow::Error::msg)
}
fn exact_bf16(actual: &MlxArray, expected: &MlxArray, name: &str) {
    assert_eq!(mx::array_dtype(actual), mx::dtype::BFLOAT16, "{name}");
    assert_eq!(mx::array_dtype(expected), mx::dtype::BFLOAT16, "{name}");
    assert_eq!(mx::array_shape(actual), mx::array_shape(expected), "{name}");
    let actual = mx::utils::array_to_vec_f32(actual);
    let expected = mx::utils::array_to_vec_f32(expected);
    assert!(actual.iter().all(|x| x.is_finite()), "{name}");
    assert_eq!(actual, expected, "{name} differs from pinned upstream BF16");
}

#[test]
fn compiled_layers_match_pinned_bf16_oracle() -> Result<()> {
    let w = layer_oracle()?;
    // The fixture records equivalence for these inputs; it does not establish
    // that eager and compiled operations always agree. Test cached calls too.
    for _ in 0..2 {
        exact_bf16(
            &*mx::compiled_objects_modulate(
                &w["modulate.x"],
                &w["modulate.scale"],
                &w["modulate.shift"],
            )?,
            &w["modulate.out"],
            "modulate",
        );
        exact_bf16(
            &*mx::compiled_objects_gated_residual(
                &w["residual.x"],
                &w["residual.h"],
                &w["residual.gate"],
            )?,
            &w["residual.out"],
            "gated residual",
        );
        exact_bf16(
            &*mx::compiled_objects_head_norm(&w["head_norm.x"], &w["head_norm.gamma"])?,
            &w["head_norm.out"],
            "head norm",
        );
    }
    Ok(())
}

#[test]
fn compiled_objects_gelu_matches_pinned_weak_scalar_oracle() -> Result<()> {
    let w = layer_oracle()?;
    let actual = mx::compiled_objects_gelu(&w["gelu.x"])?;
    exact_bf16(&actual, &w["gelu.out"], "weak scalar erf GELU");
    let promoted = mx::astype(&w["gelu.f32_constants"], mx::dtype::BFLOAT16);
    let promoted = mx::utils::array_to_vec_f32(&promoted);
    let expected = mx::utils::array_to_vec_f32(&w["gelu.out"]);
    assert_eq!(
        promoted
            .iter()
            .zip(expected)
            .filter(|(a, b)| **a != *b)
            .count(),
        3857
    );
    assert!(mx::compiled_objects_gelu(&mx::zeros(&[2], mx::dtype::INT32)).is_err());
    Ok(())
}

#[test]
fn head_norm_constructor_rejects_malformed_gamma_rank_and_shape() -> Result<()> {
    let mut w = WeightMap::new();
    w.insert("norm.gamma".into(), mx::zeros(&[2, 4], mx::dtype::BFLOAT16));
    assert!(HeadNorm::from_weights(&w, "norm", 2, 4).is_ok());
    for shape in [vec![], vec![8], vec![1, 2, 4], vec![4, 2], vec![2, 3]] {
        w.insert("norm.gamma".into(), mx::zeros(&shape, mx::dtype::BFLOAT16));
        let error = HeadNorm::from_weights(&w, "norm", 2, 4)
            .err()
            .ok_or_else(|| anyhow::anyhow!("accepted invalid gamma shape {shape:?}"))?;
        assert!(error.to_string().contains("norm.gamma"), "{error}");
    }
    assert!(HeadNorm::from_weights(&w, "norm", 0, 4).is_err());
    assert!(HeadNorm::from_weights(&w, "norm", 2, usize::MAX).is_err());
    Ok(())
}
