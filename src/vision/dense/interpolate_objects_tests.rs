// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn oracle(key: &str) -> UniquePtr<MlxArray> {
    let document: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/sam3d_objects/resize-policy-provenance.json"
    ))
    .unwrap();
    let value = &document["arrays"][key];
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect();
    let values: Vec<f32> = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect();
    mlxcel_core::from_slice_f32(&values, &shape)
}

fn bits(x: &MlxArray) -> Vec<u32> {
    mlxcel_core::utils::array_to_vec_f32(x)
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
fn objects_resize_nearest_matches_authentic_ratio_first_oracle() -> Result<()> {
    let x = oracle("nearest_input");
    let expected = oracle("nearest_output");
    let actual = resize_nearest_objects_nhwc(&x, (1, 518))?;
    assert_eq!(
        mlxcel_core::array_shape(&actual),
        mlxcel_core::array_shape(&expected)
    );
    assert_eq!(bits(&actual), bits(&expected));
    // The fixture catches the seven ratio-rounding boundaries in 84 -> 518.
    let legacy = resize_nearest_nhwc(&x, (1, 518))?;
    assert_eq!(
        bits(&legacy)
            .iter()
            .zip(bits(&expected))
            .filter(|(a, b)| **a != *b)
            .count(),
        7
    );
    Ok(())
}

#[test]
fn objects_resize_bilinear_matches_authentic_width_first_oracle() -> Result<()> {
    let x = oracle("bilinear_input");
    let expected = oracle("bilinear_output");
    let actual = resize_bilinear_objects_nhwc(&x, (7, 9))?;
    assert_eq!(
        mlxcel_core::array_shape(&actual),
        mlxcel_core::array_shape(&expected)
    );
    assert_eq!(bits(&actual), bits(&expected));
    let legacy = resize_bilinear_nhwc(&x, (7, 9), false, true)?;
    assert_ne!(bits(&legacy), bits(&expected));
    Ok(())
}

#[test]
fn objects_resize_preserves_identity_dtype_and_rejects_invalid_shapes() -> Result<()> {
    let x = mlxcel_core::zeros(&[1, 2, 3, 1], mlxcel_core::dtype::BFLOAT16);
    for y in [
        resize_nearest_objects_nhwc(&x, (2, 3))?,
        resize_bilinear_objects_nhwc(&x, (2, 3))?,
    ] {
        assert_eq!(mlxcel_core::array_dtype(&y), mlxcel_core::dtype::BFLOAT16);
        assert_eq!(mlxcel_core::array_shape(&y), [1, 2, 3, 1]);
    }
    assert!(resize_nearest_objects_nhwc(&x, (0, 3)).is_err());
    let invalid = mlxcel_core::zeros(&[2, 3, 1], mlxcel_core::dtype::FLOAT32);
    assert!(resize_bilinear_objects_nhwc(&invalid, (2, 3)).is_err());
    Ok(())
}

#[test]
fn objects_resize_bilinear_matches_authentic_height_first_and_tie_oracles() -> Result<()> {
    let x = oracle("bilinear_input");
    for (key, size) in [
        ("bilinear_height_first_output", (2, 3)),
        ("bilinear_tie_output", (5, 12)),
    ] {
        let expected = oracle(key);
        let actual = resize_bilinear_objects_nhwc(&x, size)?;
        assert_eq!(
            mlxcel_core::array_shape(&actual),
            mlxcel_core::array_shape(&expected)
        );
        assert_eq!(bits(&actual), bits(&expected), "{key}");
    }
    Ok(())
}

#[test]
fn objects_resize_nearest_preserves_nan_payload_and_input_dtype() -> Result<()> {
    let values = [f32::from_bits(0x7fc12345), -0.0, 2.0, f32::INFINITY];
    let x = mlxcel_core::from_slice_f32(&values, &[1, 1, 4, 1]);
    let y = resize_nearest_objects_nhwc(&x, (1, 8))?;
    assert_eq!(
        bits(&y),
        values
            .into_iter()
            .flat_map(|x| [x.to_bits(); 2])
            .collect::<Vec<_>>()
    );
    let x = mlxcel_core::astype(&x, mlxcel_core::dtype::BFLOAT16);
    let y = resize_nearest_objects_nhwc(&x, (1, 8))?;
    assert_eq!(mlxcel_core::array_dtype(&y), mlxcel_core::dtype::BFLOAT16);
    Ok(())
}
