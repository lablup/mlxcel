// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
fn objects_dino_default_rectangular_position_scales_round_once() {
    // Python (grid + 0.1) / 37, converted to the F32 tensor scalar only once.
    for (grid, expected) in [(49, 1.3270270824432373f32), (73, 1.9756757020950317f32)] {
        let actual = objects_position_scale(grid, 37, 0.1) as f32;
        assert_eq!(actual.to_bits(), expected.to_bits());
        let old = (grid as f32 + 0.1f32) / 37.;
        assert_ne!(old.to_bits(), actual.to_bits());
    }
}

#[test]
fn objects_dino_routes_rectangular_position_interpolation_without_changing_default() -> Result<()> {
    let tokens = filled(&[1, 49 * 73 + 1, 2], 0.);
    let mut model = tiny_model(518, 14, 1, 0);
    model.config.interpolate_offset = 0.1;
    let old = model.interpolate_pos_encoding(&tokens, 686, 1022)?;
    assert!(!model.uses_objects_policy());
    let model = model.with_objects_policy();
    assert!(model.uses_objects_policy());
    let actual = model.interpolate_pos_encoding(&tokens, 686, 1022)?;
    let positions = mlxcel_core::reshape(
        &mlxcel_core::slice(&model.pos_embed, &[0, 1, 0], &[1, 1370, 2]),
        &[1, 37, 37, 2],
    );
    let expected =
        resize_bicubic_objects_nhwc(&positions, None, Some((49.1 / 37., 73.1 / 37.)), false)?;
    let expected = mlxcel_core::reshape(&expected, &[1, 49 * 73, 2]);
    let actual_patches = mlxcel_core::slice(&actual, &[0, 1, 0], &[1, 49 * 73 + 1, 2]);
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&actual_patches),
        mlxcel_core::array_to_raw_bytes(&expected)
    );
    assert_ne!(
        mlxcel_core::array_to_raw_bytes(&actual),
        mlxcel_core::array_to_raw_bytes(&old)
    );
    Ok(())
}
